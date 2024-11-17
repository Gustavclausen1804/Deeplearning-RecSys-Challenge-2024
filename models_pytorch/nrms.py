# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models_pytorch.layers import AttLayer2, SelfAttention

class UserEncoder(nn.Module):
    def __init__(self, titleencoder, hparams, seed):
        super(UserEncoder, self).__init__()
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(
            multiheads=hparams["head_num"], 
            head_dim=hparams["head_dim"], 
            seed=seed
        )
        self.attention_layer = AttLayer2(
            dim=hparams["news_output_dim"],  # Ensure this matches the news_output_dim
            seed=seed
        )
        
        # Remove the projection layer if it's no longer needed
        # If you still need to use user_projection, ensure in_features matches the output of attention_layer
        self.user_projection = nn.Linear(
            in_features=hparams["news_output_dim"], 
            out_features=hparams["news_output_dim"]
        )

    def forward(self, his_input_title):
        batch_size, history_size, title_size = his_input_title.size()
        #print(f"UserEncoder - his_input_title shape: {his_input_title.shape}")

        # Encode each historical title
        click_title_presents = []
        for i in range(history_size):
            title = his_input_title[:, i, :]  # (batch_size, title_size)
            encoded_title = self.titleencoder(title)  # (batch_size, output_dim)
            click_title_presents.append(encoded_title)
        click_title_presents = torch.stack(click_title_presents, dim=1)  # (batch_size, history_size, output_dim)
        #print(f"UserEncoder - click_title_presents shape after encoding: {click_title_presents.shape}")

        # Apply self-attention
        y = self.self_attention([click_title_presents, click_title_presents, click_title_presents])  # (batch_size, history_size, output_dim)
        #print(f"UserEncoder - output shape after self_attention: {y.shape}")
        
        # Apply corrected attention layer
        y = self.attention_layer(y)  # (batch_size, news_output_dim)
        #print(f"UserEncoder - output shape after attention_layer: {y.shape}")

        # Apply the projection if necessary
        y = self.user_projection(y)  # Now shape will be (batch_size, news_output_dim)
        #print(f"UserEncoder - output shape after user_projection: {y.shape}")

        return y


class NewsEncoder(nn.Module):
    def __init__(self, embedding_layer, hparams, seed=42):
        super(NewsEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.output_dim = hparams.get('news_output_dim', 200)
        self.embedding = embedding_layer

        self.fc1 = nn.Linear(self.embedding.embedding_dim, hparams.get('hidden_dim', 128))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hparams.get('hidden_dim', 128), self.output_dim)

    def forward(self, x):
        x = x.long()
        
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)  # Aggregate over sequence length
        
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.fc2(out)
        
        #print(f"NewsEncoder - output shape: {out.shape}")
        return out


class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=42, device='cuda'):
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        if word2vec_embedding is None:
            embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_emb_dim, padding_idx=0)
        else:
            embedding_layer = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(word2vec_embedding).float(),
                freeze=False,
                padding_idx=0
            )
        self.word_emb_dim = embedding_layer.embedding_dim

        # Move embedding layer to specified device
        embedding_layer = embedding_layer.to(device)

        # Define NewsEncoder and UserEncoder
        self.newsencoder = self._build_newsencoder(embedding_layer).to(device)
        self.userencoder = self._build_userencoder(self.newsencoder).to(device)

    def _build_newsencoder(self, embedding_layer):
        return NewsEncoder(embedding_layer, self.hparams, self.seed)

    def _build_userencoder(self, titleencoder):
        

        return UserEncoder(titleencoder, self.hparams, self.seed)

    def forward(self, his_input_title, pred_input_title):
        # Move input tensors to the correct device
        his_input_title = his_input_title.to(self.device)
        pred_input_title = pred_input_title.to(self.device)
        
        user_present = self.userencoder(his_input_title)  # (batch_size, output_dim)
        #print(f"NRMSModel - user_present shape: {user_present.shape}")
        
        batch_size, npratio_plus_1, title_size = pred_input_title.size()

        # Encode prediction titles
        news_present = []
        for i in range(npratio_plus_1):
            title = pred_input_title[:, i, :]  # (batch_size, title_size)
            encoded_title = self.newsencoder(title)  # (batch_size, output_dim)
            news_present.append(encoded_title)
        news_present = torch.stack(news_present, dim=1)  # (batch_size, npratio_plus_1, output_dim)
        #print(f"NRMSModel - news_present shape after encoding: {news_present.shape}")

        # Compute scores using dot product
        scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)  # (batch_size, npratio_plus_1)
        #print(f"NRMSModel - scores shape: {scores.shape}")

        return scores  # (batch_size, npratio_plus_1)

    def predict(self, his_input_title, pred_input_title_one):
        # Move input tensors to the correct device
        his_input_title = his_input_title.to(self.device)
        pred_input_title_one = pred_input_title_one.to(self.device)
        
        user_present = self.userencoder(his_input_title)  # (batch_size, output_dim)
        news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))  # (batch_size, output_dim)
        
        score = torch.sum(news_present_one * user_present, dim=1)  # (batch_size,)
        #print(f"NRMSModel - predict score shape: {score.shape}")

        return torch.sigmoid(score)  # (batch_size,)

    def get_loss(self, criterion="cross_entropy"):
        if criterion == "cross_entropy":
            return nn.CrossEntropyLoss().to(self.device)
        elif criterion == "log_loss":
            return nn.BCELoss().to(self.device)
        else:
            raise ValueError(f"Loss function not defined: {criterion}")

    def get_optimizer(self, optimizer="adam", lr=1e-3):
        if optimizer == "adam":
            return optim.Adam(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Optimizer not defined: {optimizer}")
