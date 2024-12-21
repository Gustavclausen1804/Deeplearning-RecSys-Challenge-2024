import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp

from models_pytorch.layers import SelfAttention

class TimeGate(nn.Module):
    def __init__(self, hidden_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        ).to(device)
    
    def forward(self, embeddings, timestamps):
        # Concatenate embeddings with timestamps
        gate_input = torch.cat([embeddings, timestamps], dim=-1)
        gates = self.gate_network(gate_input)
        return embeddings * gates

class NewsEncoderDocVec(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super().__init__()
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        head_num = hparams["head_num"]
        head_dim = hparams["head_dim"]
        self.output_dim = head_num * head_dim
        
        self.units_per_layer = hparams.get('units_per_layer', [512, 512, 512])
        input_dim = hparams.get('embedding_dim', 30)
        
        # Create layers with residual connections
        self.layers = nn.ModuleList()
        for units in self.units_per_layer:
            block = nn.Sequential(
                nn.Linear(input_dim, units),
                nn.ReLU(),
                nn.LayerNorm(units),
                nn.Dropout(hparams.get('dropout', 0.2))
            ).to(device)
            self.layers.append(block)
            
            if input_dim != units:
                self.layers.append(nn.Linear(input_dim, units).to(device))
            input_dim = units

        self.final = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU()
        ).to(device)

    def forward(self, x):
        x = x.to(self.device)
        
        if len(x.shape) == 3:
            batch_size, num_titles, docvec_dim = x.shape
            x = x.view(-1, docvec_dim)
        else:
            batch_size = None
            num_titles = None

        for i in range(0, len(self.layers), 2):
            identity = x
            x = self.layers[i](x)
            
            if i+1 < len(self.layers):
                identity = self.layers[i+1](identity)
            
            x = x + identity

        out = self.final(x)

        if batch_size is not None and num_titles is not None:
            out = out.view(batch_size, num_titles, -1)

        return out

class TimeAwareAttention(nn.Module):
    def __init__(self, hidden_dim, device='cuda'):
        super().__init__()
        self.device = device
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, encoded_titles, timestamps):
        # Process time information
        time_features = self.time_mlp(timestamps)
        
        # Combine content and time features
        combined_features = torch.cat([encoded_titles, time_features], dim=-1)
        
        # Compute attention weights
        attention_weights = self.attention(combined_features)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended_output = torch.sum(encoded_titles * attention_weights, dim=1)
        return attended_output

class UserEncoderDocVec(nn.Module):
    def __init__(self, news_encoder, hparams, seed, device='cuda'):
        super().__init__()
        self.device = device
        self.news_encoder = news_encoder
        
        hidden_dim = hparams["head_num"] * hparams["head_dim"]
        
        self.time_gate = TimeGate(hidden_dim, device)
        self.self_attention = SelfAttention(
            multiheads=hparams["head_num"],
            head_dim=hparams["head_dim"],
            seed=seed,
            device=device
        )
        self.time_aware_attention = TimeAwareAttention(
            hidden_dim=hidden_dim,
            device=device
        )

        # Time embedding module
        # We choose a small MLP that maps a scalar time delta to the same dimension as articles.
        time_embedding_dim = hparams["head_num"] * hparams["head_dim"]
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.ReLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        ).to(device)

    def forward(self, his_input_title, his_input_time):
        his_input_title = his_input_title.to(self.device)
        his_input_time = his_input_time.to(self.device).unsqueeze(-1)

        # Encode titles
        encoded_titles = self.news_encoder(his_input_title)
        
        # Apply time gating to modulate features
        gated_titles = self.time_gate(encoded_titles, his_input_time)
        
        # Self-attention to capture relationships
        attended_features = self.self_attention([gated_titles, gated_titles, gated_titles])
        
        # Final time-aware attention for aggregation
        user_vector = self.time_aware_attention(attended_features, his_input_time)
        
        return user_vector

class NRMSDocVecModel(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super().__init__()
        self.hparams = hparams
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(self.newsencoder)

        self.to(self.device)
        self.scaler = amp.GradScaler()

    def _build_newsencoder(self):
        return NewsEncoderDocVec(self.hparams, self.seed, self.device)

    def _build_userencoder(self, news_encoder):
        return UserEncoderDocVec(news_encoder, self.hparams, self.seed, self.device)

    def forward(self, his_input_title, his_input_time, pred_input_title):
        with torch.cuda.amp.autocast(enabled=False):
            his_input_title = his_input_title.to(self.device)
            his_input_time = his_input_time.to(self.device)
            pred_input_title = pred_input_title.to(self.device)

            user_present = self.userencoder(his_input_title, his_input_time)
            news_present = self.newsencoder(pred_input_title)

            if len(news_present.shape) == 2:
                batch_size = his_input_title.size(0)
                num_titles = pred_input_title.size(1)
                news_present = news_present.view(batch_size, num_titles, -1)

            scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
            return scores

    def score(self, his_input_title, his_input_time, pred_input_title_one):
        with torch.no_grad():
            user_present = self.userencoder(his_input_title, his_input_time)
            news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))
            scores = torch.sum(news_present_one * user_present, dim=1, keepdim=True)
            return torch.sigmoid(scores)

    def get_loss(self, criterion="cross_entropy"):
        if criterion == "cross_entropy":
            return nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        elif criterion == "log_loss":
            return nn.BCELoss().to(self.device)
        else:
            raise ValueError(f"Loss function not defined: {criterion}")

    def get_optimizer(self, hparams_nrms):
        print('learning rate: ', hparams_nrms.__dict__['learning_rate'])
        if hparams_nrms.__dict__['optimizer'] == "adam":
            return optim.Adam(
                self.parameters(),
                lr=hparams_nrms.__dict__['learning_rate'],
                weight_decay=hparams_nrms.__dict__['weight_decay']
            )
        else:
            raise ValueError(f"Optimizer not defined")

    def predict(self, his_input_title, his_input_time, pred_input_title):
        with torch.no_grad():
            scores = self.forward(his_input_title, his_input_time, pred_input_title)
            return torch.sigmoid(scores)