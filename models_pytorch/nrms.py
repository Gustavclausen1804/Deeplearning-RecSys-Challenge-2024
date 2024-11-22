# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp

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
        reshaped_input = his_input_title.view(-1, title_size)
        encoded_titles = self.titleencoder(reshaped_input)
        click_title_presents = encoded_titles.view(batch_size, history_size, -1)
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
    def __init__(self, hparams, word2vec_embedding=None, word_emb_dim=64, vocab_size=10000, seed=42, device='cuda', fos=2):
        super(NRMSModel, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        if word2vec_embedding is None:
            embedding_layer = nn.Embedding(
                num_embeddings=vocab_size,  # Reduced from 30522
                embedding_dim=word_emb_dim, # Reduced from 128
                padding_idx=0
            )
        else:
            embedding_layer = nn.Embedding.from_pretrained(
                embeddings=torch.from_numpy(word2vec_embedding).float(),
                freeze=True,
            )

        # Define NewsEncoder and UserEncoder with device
        self.newsencoder = self._build_newsencoder(embedding_layer)
        self.userencoder = self._build_userencoder(self.newsencoder)
        
        # Move entire model to device
        self.to(self.device)
        self.scaler = amp.GradScaler()
        self.embedding_cache = {}  # Cache for embeddings during inference

    def _build_newsencoder(self, embedding_layer):
        return NewsEncoder(embedding_layer, self.hparams, self.seed, self.device)

    def _build_userencoder(self, titleencoder):
        return UserEncoder(titleencoder, self.hparams, self.seed, self.device)

    def _batch_encode_news(self, titles):
        """Batch process all titles at once"""
        batch_size, num_titles, title_size = titles.size()
        titles_flat = titles.view(-1, title_size)
        encoded_flat = self.newsencoder(titles_flat)
        return encoded_flat.view(batch_size, num_titles, -1)

    def forward(self, his_input_title, pred_input_title):
        with torch.cuda.amp.autocast():
            his_input_title = his_input_title.to(self.device)
            pred_input_title = pred_input_title.to(self.device)
            
            user_present = self.userencoder(his_input_title)
            news_present = self._batch_encode_news(pred_input_title)
            
            # Compute scores using batch matrix multiplication
            scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
            
            return scores

    def _cache_key(self, tensor):
        return hash(tensor.cpu().numpy().tobytes())

    @torch.no_grad()
    def predict(self, dataloader):
        self.eval()
        all_predictions = []
        
        for batch in dataloader:
            his_input_title, pred_input_title = batch[0]
            
            with amp.autocast():
                # Move to device and process in batches
                his_input_title = his_input_title.to(self.device)
                pred_input_title = pred_input_title.to(self.device)
                
                # Get user embeddings for whole batch
                user_present = self.userencoder(his_input_title)
                
                # Process all candidates at once
                news_present = self._batch_encode_news(pred_input_title)
                
                # Compute all scores at once using batch matrix multiplication
                scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
                
                # Split scores by user
                for user_scores in scores:
                    # Only keep scores for non-padding candidates
                    mask = (pred_input_title[0] != 0).any(dim=-1)
                    valid_scores = user_scores[mask].cpu().tolist()
                    all_predictions.append(valid_scores)
        
        return all_predictions

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
        self.eval()
        all_predictions = []
        
        for batch in dataloader:
            his_input_title, pred_input_title = batch[0]
            batch_size = his_input_title.size(0)
            
            # Move to device
            his_input_title = his_input_title.to(self.device)
            pred_input_title = pred_input_title.to(self.device)
            
            # Get user embeddings
            user_present = self.userencoder(his_input_title)
            
            # Get predictions for each user in batch
            for i in range(batch_size):
                user_scores = []
                user_emb = user_present[i:i+1]
                n_candidates = (pred_input_title[i] != 0).any(dim=-1).sum().item()  # Count non-zero rows
                
                # Only predict for non-padding candidates
                for j in range(n_candidates):
                    title = pred_input_title[i:i+1, j, :]
                    news_present = self.newsencoder(title)
                    score = torch.sum(news_present * user_emb, dim=1)
                    user_scores.append(score.item())
                
                all_predictions.append(user_scores)
        
        return all_predictions



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