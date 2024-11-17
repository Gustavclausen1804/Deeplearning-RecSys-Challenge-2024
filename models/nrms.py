import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from models.layers import AttLayer2, SelfAttention

class NRMSModel(nn.Module):
    def __init__(
        self,
        hparams: dict,
        word2vec_embedding: np.ndarray = None,
        word_emb_dim: int = 300,
        vocab_size: int = 32000,
        seed: int = None,
    ):
        super().__init__()
        self.hparams = hparams
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize word embeddings
        if word2vec_embedding is None:
            self.word2vec_embedding = nn.Embedding(vocab_size, word_emb_dim)
        else:
            self.word2vec_embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(word2vec_embedding),
                freeze=False
            )
        
        # Build encoder components
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder()
        
        # Similarity computation
        self.dot_product = nn.Linear(self.hparams.attention_hidden_dim, self.hparams.attention_hidden_dim, bias=False)
    
    def _build_newsencoder(self):
        return nn.Sequential(
            self.word2vec_embedding,
            nn.Dropout(self.hparams.dropout),
            SelfAttention(self.hparams.head_num, self.hparams.head_dim),
            nn.Dropout(self.hparams.dropout),
            AttLayer2(self.hparams.attention_hidden_dim)
        )
    
    def _build_userencoder(self):
        return nn.Sequential(
            SelfAttention(self.hparams.head_num, self.hparams.head_dim),
            AttLayer2(self.hparams.attention_hidden_dim)
        )
    
    def get_user_embedding(self, clicked_news):
        """
        Args:
            clicked_news: tensor of shape [batch_size, history_size, title_size]
                         or [batch_size, 1, history_size, title_size]
        """
        if len(clicked_news.shape) == 4:
            clicked_news = clicked_news.squeeze(1)
        
        batch_size, history_size, title_size = clicked_news.shape
        # Reshape for processing each title
        clicked_news_reshape = clicked_news.view(-1, title_size)
        
        # Get embeddings for each title
        news_embeddings = self.newsencoder(clicked_news_reshape)
        # Reshape back
        news_embeddings = news_embeddings.view(batch_size, history_size, -1)
        
        return self.userencoder(news_embeddings)
    
    def get_news_embedding(self, title):
        """
        Args:
            title: tensor of shape [batch_size, npratio, title_size]
                  or [batch_size, 1, npratio, title_size]
        """
        if len(title.shape) == 4:
            title = title.squeeze(1)
            
        batch_size, npratio, title_size = title.shape
        # Reshape for processing each title
        title_reshape = title.view(-1, title_size)
        
        # Get embeddings
        embeddings = self.newsencoder(title_reshape)
        # Reshape back
        return embeddings.view(batch_size, npratio, -1)
    
    def forward(self, his_input_title, pred_input_title, compute_scores=True):
        """
        Args:
            his_input_title: tensor of shape [batch_size, 1, history_size, title_size]
            pred_input_title: tensor of shape [batch_size, 1, npratio, title_size]
        """
        print(f"\nInput shapes in forward pass:")
        print(f"his_input_title shape: {his_input_title.shape}")
        print(f"pred_input_title shape: {pred_input_title.shape}")
        
        user_present = self.get_user_embedding(his_input_title)
        news_present = self.get_news_embedding(pred_input_title)
        
        if compute_scores:
            # Compute similarity scores
            scores = torch.matmul(news_present, user_present.unsqueeze(-1)).squeeze(-1)
            return F.softmax(scores, dim=-1)
        else:
            return user_present, news_present
        
    def scoring(self, his_input_title, pred_input_title_one):
        """
        Args:
            his_input_title: tensor of shape [batch_size, 1, history_size, title_size]
            pred_input_title_one: tensor of shape [batch_size, 1, 1, title_size]
        """
        user_present = self.get_user_embedding(his_input_title)
        news_present_one = self.get_news_embedding(pred_input_title_one)
        
        scores = torch.matmul(news_present_one, user_present.unsqueeze(-1)).squeeze(-1)
        return torch.sigmoid(scores)
    
    def configure_optimizers(self, optimizer='adam', learning_rate=0.001):
        if optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=learning_rate)
        raise ValueError(f"Optimizer {optimizer} not supported")
    
    def get_loss_fn(self, loss_type):
        if loss_type == "cross_entropy_loss":
            return nn.CrossEntropyLoss()
        elif loss_type == "log_loss":
            return nn.BCELoss()
        raise ValueError(f"Loss {loss_type} not supported")
    
    def predict(self, dataloader):
        """
        Make predictions using the model
        Args:
            dataloader: DataLoader containing validation/test data
        Returns:
            numpy array of predictions
        """
        self.eval()  # Set model to evaluation mode
        device = next(self.parameters()).device
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc='Predicting'):
                inputs = [x.to(device) for x in inputs]
                scores = self.scoring(*inputs)  # Use the scoring method we defined earlier
                predictions.append(scores.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)

class NRMSTrainer:
    def __init__(self, model, optimizer, loss_fn, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train_step(self, batch):
        self.model.train()
        his_title, pred_title = [x.to(self.device) for x in batch[0]]
        labels = batch[1].to(self.device)
        
        self.optimizer.zero_grad()
        scores = self.model(his_title, pred_title)
        loss = self.loss_fn(scores, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, batch):
        self.model.eval()
        with torch.no_grad():
            his_title, pred_title = [x.to(self.device) for x in batch[0]]
            labels = batch[1].to(self.device)
            scores = self.model(his_title, pred_title)
            loss = self.loss_fn(scores, labels)
        return loss.item()