# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models_pytorch.layers import AttLayer2, SelfAttention

class UserEncoder(nn.Module):
    def __init__(self, titleencoder, hparams, seed, device='cuda'):
        super(UserEncoder, self).__init__()
        self.device = device
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(
            multiheads=hparams["head_num"], 
            head_dim=hparams["head_dim"], 
            seed=seed,
            device=device
        )
        self.attention_layer = AttLayer2(
            dim=hparams["news_output_dim"],
            seed=seed,
            device=device
        ).to(device)
        self.user_projection = nn.Linear(
            in_features=hparams["news_output_dim"], 
            out_features=hparams["news_output_dim"]
        ).to(device)

    def forward(self, his_input_title):
        his_input_title = his_input_title.to(self.device)
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
    def __init__(self, embedding_layer, hparams, seed=42, device='cuda'):
        super(NewsEncoder, self).__init__()
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.output_dim = hparams.get('news_output_dim', 200)
        self.embedding = embedding_layer.to(device)
        self.fc1 = nn.Linear(self.embedding.embedding_dim, hparams.get('hidden_dim', 128)).to(device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hparams.get('hidden_dim', 128), self.output_dim).to(device)

    def forward(self, x):
        x = x.to(self.device)
        x = x.long()
        
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)  # Aggregate over sequence length
        
        out = self.fc1(embedded)
        out = self.relu(out)
        out = self.fc2(out)
        
        #print(f"NewsEncoder - output shape: {out.shape}")
        return out


class NRMSModel(nn.Module):
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=300, vocab_size=32000, seed=42, device='cuda', fos=2):
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
        embedding_layer = embedding_layer.to(device)

        # Define NewsEncoder and UserEncoder with device
        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)
        
        # Move entire model to device
        self.to(self.device)

    def _build_newsencoder(self, embedding_layer):
        return NewsEncoder(embedding_layer, self.hparams, self.seed, self.device)

    def _build_userencoder(self, titleencoder):
        return UserEncoder(titleencoder, self.hparams, self.seed, self.device)

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

    def score(self, his_input_title, pred_input_title_one):
        """Equivalent to TF scorer model for evaluation
        Args:
            his_input_title: tensor of shape (batch_size, history_size, title_size)
            pred_input_title_one: tensor of shape (batch_size, 1, title_size)
        Returns:
            scores: tensor of shape (batch_size, 1)
        """
        # Move input tensors to the correct device
        his_input_title = his_input_title.to(self.device)
        pred_input_title_one = pred_input_title_one.to(self.device)
        
        # Get user representation
        user_present = self.userencoder(his_input_title)
        
        # Get news representation for single item
        news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))
        
        # Compute dot product and apply sigmoid
        scores = torch.sum(news_present_one * user_present, dim=1, keepdim=True)
        return torch.sigmoid(scores)

    @torch.no_grad()
    def predict(self, dataloader):
        """Predict scores for validation data handling variable batch sizes"""
        self.eval()
        all_scores = []
        print("\nStarting prediction...")
        
        for batch_idx, batch in enumerate(dataloader):
            his_input_title, pred_input_title = batch[0]
            his_input_title = his_input_title.to(self.device)
            pred_input_title = pred_input_title.to(self.device)
            
            # Process entire batch at once
            batch_size = his_input_title.size(0)
            user_present = self.userencoder(his_input_title)
            
            # Process each sample in the batch individually
            for i in range(batch_size):
                sample_scores = []
                user_emb = user_present[i:i+1]  # Keep batch dimension
                pred_titles = pred_input_title[i]  # Get all predictions for this sample
                
                # Get scores for each candidate article
                for j in range(pred_titles.size(0)):
                    pred_one = pred_titles[j:j+1].unsqueeze(0)  # Add batch dimension
                    news_present_one = self.newsencoder(pred_one.squeeze(1))
                    score = torch.sum(news_present_one * user_emb, dim=1)
                    sample_scores.append(score.item())
                
                all_scores.append(sample_scores)
        
        return all_scores

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
