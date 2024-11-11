# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models_pytorch.layers import AttLayer2, SelfAttention

class NewsEncoder(nn.Module):
    def __init__(self, embedding_layer, hparams, seed=42):
        """
        News Encoder Module.

        Args:
            embedding_layer (nn.Embedding): Pretrained embedding layer.
            hparams (dict): Hyperparameters dictionary.
            seed (int): Random seed for reproducibility.
        """
        super(NewsEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Define output dimension from hyperparameters or default
        self.output_dim = hparams.get('news_output_dim', 200)

        # Use the provided embedding layer
        self.embedding = embedding_layer  # nn.Embedding

        # Define the rest of the encoder layers
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hparams.get('hidden_dim', 128))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hparams.get('hidden_dim', 128), self.output_dim)

    def forward(self, x):
        """
        Forward pass of the NewsEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, output_dim).
        """
        # Ensure x is of type LongTensor for embedding
        x = x.long()
        
        # Obtain embeddings: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # Aggregate embeddings by taking the mean over the sequence length
        embedded = embedded.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Pass through fully connected layers
        out = self.fc1(embedded)          # (batch_size, hidden_dim)
        out = self.relu(out)
        out = self.fc2(out)               # (batch_size, output_dim)
        
        return out


class NRMSModel(nn.Module):
    """
    NRMS Model (Neural News Recommendation with Multi-Head Self-Attention).

    Reference:
    Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, and Xing Xie, 
    "Neural News Recommendation with Multi-Head Self-Attention" 
    in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing 
    and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)
    """

    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = 42,
    ):
        """
        Initializes the NRMSModel.

        Args:
            hparams (dict): Hyperparameters dictionary.
            word2vec_embedding (np.ndarray, optional): Pretrained word embeddings. Defaults to None.
            word_emb_dim (int, optional): Dimension of word embeddings. Defaults to 300.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 32000.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize word embeddings
        if word2vec_embedding is None:
            # Initialize embedding layer randomly
            embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_emb_dim, padding_idx=0)
        else:
            # Initialize embedding layer with pre-trained word2vec embeddings
            embedding_layer = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(word2vec_embedding).float(),
                freeze=False,  # Allow fine-tuning
                padding_idx=0   # Assuming 0 is the padding index
            )
        self.word_emb_dim = embedding_layer.embedding_dim

        # Define NewsEncoder
        self.newsencoder = self._build_newsencoder(embedding_layer)

        # Define UserEncoder
        self.userencoder = self._build_userencoder(self.newsencoder)

    def _build_newsencoder(self, embedding_layer):
        """
        Builds the NewsEncoder component.

        Args:
            embedding_layer (nn.Embedding): Embedding layer.

        Returns:
            NewsEncoder: Initialized NewsEncoder module.
        """
        return NewsEncoder(embedding_layer, self.hparams, self.seed)

    def _build_userencoder(self, titleencoder):
        """
        Builds the UserEncoder component.

        Args:
            titleencoder (NewsEncoder): NewsEncoder module.

        Returns:
            nn.Module: Initialized UserEncoder module.
        """
        
        class UserEncoder(nn.Module):
            def __init__(self, titleencoder, hparams, seed):
                """
                User Encoder Module.

                Args:
                    titleencoder (NewsEncoder): NewsEncoder module.
                    hparams (dict): Hyperparameters dictionary.
                    seed (int): Random seed for reproducibility.
                """
                super(UserEncoder, self).__init__()
                self.titleencoder = titleencoder
                self.self_attention = SelfAttention(
                    multiheads=hparams["head_num"], 
                    head_dim=hparams["head_dim"], 
                    seed=seed
                )
                self.attention_layer = AttLayer2(
                    dim=hparams["attention_hidden_dim"], 
                    seed=seed
                )

            def forward(self, his_input_title):
                """
                Forward pass of the UserEncoder.

                Args:
                    his_input_title (torch.Tensor): Historical input titles tensor of shape (batch_size, history_size, title_size).

                Returns:
                    torch.Tensor: Aggregated user representation tensor of shape (batch_size, attention_hidden_dim).
                """
                batch_size, history_size, title_size = his_input_title.size()
                device = his_input_title.device

                # Encode each historical title
                click_title_presents = []
                for i in range(history_size):
                    title = his_input_title[:, i, :]  # (batch_size, title_size)
                    encoded_title = self.titleencoder(title)  # (batch_size, output_dim)
                    click_title_presents.append(encoded_title)
                click_title_presents = torch.stack(click_title_presents, dim=1)  # (batch_size, history_size, output_dim)

                # Apply self-attention
                y = self.self_attention([click_title_presents, click_title_presents, click_title_presents])  # (batch_size, history_size, output_dim)
                
                # Apply attention layer to aggregate
                y = self.attention_layer(y)  # (batch_size, output_dim)

                return y

        return UserEncoder(titleencoder, self.hparams, self.seed)

    def forward(self, his_input_title, pred_input_title):
        """
        Forward pass of the NRMSModel.

        Args:
            his_input_title (torch.Tensor): Historical input titles tensor of shape (batch_size, history_size, title_size).
            pred_input_title (torch.Tensor): Prediction input titles tensor of shape (batch_size, npratio_plus_1, title_size).

        Returns:
            torch.Tensor: Output probabilities of shape (batch_size, npratio_plus_1).
        """
        user_present = self.userencoder(his_input_title)  # (batch_size, output_dim)
        batch_size, npratio_plus_1, title_size = pred_input_title.size()
        device = pred_input_title.device

        # Encode prediction titles
        news_present = []
        for i in range(npratio_plus_1):
            title = pred_input_title[:, i, :]  # (batch_size, title_size)
            encoded_title = self.newsencoder(title)  # (batch_size, output_dim)
            news_present.append(encoded_title)
        news_present = torch.stack(news_present, dim=1)  # (batch_size, npratio_plus_1, output_dim)

        # Compute scores using dot product
        # user_present: (batch_size, output_dim)
        # news_present: (batch_size, npratio_plus_1, output_dim)
        # To compute scores: (batch_size, npratio_plus_1)
        scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)  # (batch_size, npratio_plus_1)

        # Apply softmax over npratio_plus_1
        return torch.softmax(scores, dim=-1)  # (batch_size, npratio_plus_1)

    def predict(self, his_input_title, pred_input_title_one):
        """
        Prediction method for a single prediction title.

        Args:
            his_input_title (torch.Tensor): Historical input titles tensor of shape (batch_size, history_size, title_size).
            pred_input_title_one (torch.Tensor): Prediction input title tensor of shape (batch_size, 1, title_size).

        Returns:
            torch.Tensor: Output probability of shape (batch_size,).
        """
        user_present = self.userencoder(his_input_title)  # (batch_size, output_dim)
        news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))  # (batch_size, output_dim)
        
        # Compute dot product between user and single news representation
        score = torch.sum(news_present_one * user_present, dim=1)  # (batch_size,)

        return torch.sigmoid(score)  # (batch_size,)

    def get_loss(self, criterion="cross_entropy"):
        """
        Returns the specified loss function.

        Args:
            criterion (str, optional): Type of loss function. Defaults to "cross_entropy".

        Returns:
            nn.Module: Loss function.
        """
        if criterion == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif criterion == "log_loss":
            return nn.BCELoss()
        else:
            raise ValueError(f"Loss function not defined: {criterion}")

    def get_optimizer(self, optimizer="adam", lr=1e-3):
        """
        Returns the specified optimizer.

        Args:
            optimizer (str, optional): Type of optimizer. Defaults to "adam".
            lr (float, optional): Learning rate. Defaults to 1e-3.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        if optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer not defined: {optimizer}")
