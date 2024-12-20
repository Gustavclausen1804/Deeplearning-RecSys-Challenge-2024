import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp
from torch.nn import LayerNorm

from models_pytorch.layers import AttLayer2, SelfAttention

class DocEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', dropout=0.2, seed=42):
        super(DocEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout),
            
        ).to(device)

    def forward(self, x):
        # x: [batch_size, num_titles, input_dim]
        batch_size, num_titles, _ = x.size()
        out = self.layer(x.view(-1, x.size(-1)))
        out = out.view(batch_size, num_titles, -1)
        return out

class CategoryEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', dropout=0.2, seed=42):
        super(CategoryEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device

        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout)
        ).to(device)
        
    def forward(self, x):
        batch_size, num_titles, _ = x.size()
        out = self.layer(x.view(-1, x.size(-1)))
        out = out.view(batch_size, num_titles, -1)
        return out

class TopicEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', dropout=0.2, seed=42):
        super(TopicEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device

        self.layer = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout)
        ).to(device)
        
    def forward(self, x):
        batch_size, num_titles, _ = x.size()
        out = self.layer(x.view(-1, x.size(-1)))
        out = out.view(batch_size, num_titles, -1)
        return out

class NumericEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, device='cuda', dropout=0.2, seed=42):
        super(NumericEncoder, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = device

        # Redesigned NumericEncoder with more layers
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(dropout)
        ).to(device)
        
    def forward(self, x):
        batch_size, num_titles, _ = x.size()
        out = self.layer(x.view(-1, x.size(-1)))
        out = out.view(batch_size, num_titles, -1)
        return out

class TimeDiscount(nn.Module):
    def __init__(self):
        super(TimeDiscount, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, delta_time):
        return self.layer(delta_time.unsqueeze(-1)).squeeze(-1)

class FeatureFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(FeatureFusion, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.gates = nn.ModuleList([nn.Linear(dim, output_dim) for dim in input_dims])
        self.activation = nn.Sigmoid()

    def forward(self, features):
        # Flatten features to (batch_size * num_titles, dim)
        features = [feature.view(-1, feature.size(-1)) for feature in features]
        
        transformed_features = [linear(feature) for linear, feature in zip(self.linears, features)]
        gates = [self.activation(gate(feature)) for gate, feature in zip(self.gates, features)]
        gated_features = [gate * transformed_feature for gate, transformed_feature in zip(gates, transformed_features)]
        fused_feature = sum(gated_features)
        return fused_feature

class NewsEncoderDocVec(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super(NewsEncoderDocVec, self).__init__()
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.output_dim = hparams.get('news_output_dim', 200)
        dropout = hparams.get('dropout', 0.2)

        # Base dimensions
        self.docvec_dim = hparams.get('title_size', 30)
        self.use_category = hparams.get('use_category', True)
        self.use_topic = hparams.get('use_topic', True)
        self.use_numeric = hparams.get('use_numeric', True)

        self.category_emb_dim = hparams.get("category_emb_dim", 128) if self.use_category else 0
        self.topic_emb_dim = hparams.get("topic_emb_dim", 128) if self.use_topic else 0
        self.numeric_feature_dim = 3 if self.use_numeric else 0

        doc_out_dim = hparams.get('doc_out_dim', 128)
        cat_out_dim = hparams.get('cat_out_dim', 128) if self.use_category else 0
        top_out_dim = hparams.get('top_out_dim', 128) if self.use_topic else 0
        num_out_dim = hparams.get('numeric_proj_dim', 16) if self.use_numeric else 0

        self.doc_encoder = DocEncoder(self.docvec_dim, doc_out_dim, device=self.device, dropout=dropout, seed=seed)
        
        if self.use_category:
            self.category_encoder = CategoryEncoder(self.category_emb_dim, cat_out_dim, device=self.device, dropout=dropout, seed=seed)
        if self.use_topic:
            self.topic_encoder = TopicEncoder(self.topic_emb_dim, top_out_dim, device=self.device, dropout=dropout, seed=seed)
        if self.use_numeric:
            self.numeric_encoder = NumericEncoder(self.numeric_feature_dim, num_out_dim, device=self.device, dropout=dropout, seed=seed)

        input_dims = [doc_out_dim]
        if self.use_category:
            input_dims.append(cat_out_dim)
        if self.use_topic:
            input_dims.append(top_out_dim)
        if self.use_numeric:
            input_dims.append(num_out_dim)

        fusion_output_dim = self.output_dim  # Output dimension after feature fusion
        self.feature_fusion = FeatureFusion(input_dims, fusion_output_dim).to(self.device)

        # Implement Layer Normalization
        self.layer_norm = LayerNorm(fusion_output_dim).to(self.device)

        # Time discount as a learnable function
        self.use_publication_discount = hparams.get('use_publication_discount', True)
        if self.use_publication_discount:
            self.time_discount = TimeDiscount().to(self.device)

    def forward(self, 
                docvecs, category_emb, topic_emb, 
                sentiment_scores, read_times, pageviews, 
                impression_timestamps, pred_timestamps):
        # Compute embeddings
        doc_out = self.doc_encoder(docvecs)
        cat_out = self.category_encoder(category_emb) if self.use_category else None
        top_out = self.topic_encoder(topic_emb) if self.use_topic else None

        num_out = None
        if self.use_numeric:
            numeric_features = torch.cat([
                sentiment_scores.unsqueeze(-1),
                read_times.unsqueeze(-1),
                pageviews.unsqueeze(-1)
            ], dim=-1).to(self.device)
            num_out = self.numeric_encoder(numeric_features)

        features = [doc_out]
        # # if cat_out is not None:
        # #     features.append(cat_out)
        # # if top_out is not None:
        # #     features.append(top_out)
        # # if num_out is not None:
        # #     features.append(num_out)

        batch_size, num_titles, _ = docvecs.size()

        # # Apply feature fusion
        fused_features = self.feature_fusion(features)

        # Apply layer normalization
        out = self.layer_norm(fused_features)
        out = out.view(batch_size, num_titles, -1)

        # Publication-time discount
        # if self.use_publication_discount and pred_timestamps is not None and impression_timestamps is not None:
        impression_timestamps = impression_timestamps.to(self.device).float().unsqueeze(-1)  # [batch_size,1]
        delta_news = impression_timestamps - pred_timestamps.to(self.device).float()  # [batch_size, num_titles]
        delta_news = torch.clamp(delta_news, min=0, max=100)  # Clamping to 100 hours

        # Apply time discount function
        discount_factors = self.time_discount(delta_news)
        discount_factors = discount_factors.unsqueeze(-1)  # [batch_size, num_titles, 1]
        out = out * discount_factors

        return out

class UserEncoderDocVec(nn.Module):
    def __init__(self, titleencoder, hparams, seed, device='cuda'):
        super(UserEncoderDocVec, self).__init__()
        self.device = device
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(
            num_heads=hparams["head_num"], 
            embed_dim=hparams["head_dim"], 
            seed=seed,
            device=device
        )
        self.attention_layer = AttLayer2(
            dim=hparams["news_output_dim"],
            hidden_dim=hparams["attention_hidden_dim"],
            device=device
        ).to(device)
        self.user_projection = nn.Linear(
            in_features=hparams["attention_hidden_dim"], 
            out_features=hparams["news_output_dim"]
        ).to(device)

        # Implement Layer Normalization
        self.layer_norm = LayerNorm(hparams["news_output_dim"]).to(device)

        # Time discount as a learnable function
        self.use_session_discount = hparams.get('use_session_discount', True)
        if self.use_session_discount:
            self.time_discount = TimeDiscount().to(self.device)

    def forward(self, 
                his_input_titles_padded, his_category_emb_padded, his_topic_emb_padded,
                his_sentiment_padded, his_read_times_padded, his_pageviews_padded,
                impression_timestamps, his_timestamps_padded):

        encoded_titles = self.titleencoder(
            his_input_titles_padded,
            his_category_emb_padded,
            his_topic_emb_padded,
            his_sentiment_padded,
            his_read_times_padded,
            his_pageviews_padded,
            impression_timestamps,
            his_timestamps_padded
        )  # [batch_size, history_size, output_dim]

        impression_timestamps = impression_timestamps.to(self.device).float().unsqueeze(-1) 
        delta_user = impression_timestamps - his_timestamps_padded.to(self.device).float() 
        delta_user = torch.clamp(delta_user, min=0, max=100)  # Clamping to 100 hours

        # Apply time discount function
        discount_factors = self.time_discount(delta_user)
        discount_factors = discount_factors.unsqueeze(-1)      # [batch_size, history_size, 1]
        encoded_titles = encoded_titles * discount_factors

        y = self.self_attention(encoded_titles, encoded_titles, encoded_titles)
        y = self.attention_layer(y)
        y = self.user_projection(y)

        # Apply layer normalization
        y = self.layer_norm(y)

        return y

class NRMSDocVecModel(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super(NRMSDocVecModel, self).__init__()
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

    def _build_userencoder(self, titleencoder):
        return UserEncoderDocVec(titleencoder, self.hparams, self.seed, self.device)

    def forward(self,
                his_input_titles_padded,
                his_category_emb_padded,
                his_topic_emb_padded,
                his_sentiment_padded,
                his_read_times_padded,
                his_pageviews_padded,
                his_timestamps_padded,
                pred_input_titles_padded,
                pred_category_emb_padded,
                pred_topic_emb_padded,
                pred_sentiment_padded,
                pred_read_times_padded,
                pred_pageviews_padded,
                pred_timestamps_padded,
                impression_timestamps):

        with torch.cuda.amp.autocast(enabled=False):
            user_present = self.userencoder(
                his_input_titles_padded,
                his_category_emb_padded,
                his_topic_emb_padded,
                his_sentiment_padded,
                his_read_times_padded,
                his_pageviews_padded,
                impression_timestamps=impression_timestamps,
                his_timestamps_padded=his_timestamps_padded
            )

            news_present = self.newsencoder(
                pred_input_titles_padded,
                pred_category_emb_padded,
                pred_topic_emb_padded,
                pred_sentiment_padded,
                pred_read_times_padded,
                pred_pageviews_padded,
                impression_timestamps=impression_timestamps,
                pred_timestamps=pred_timestamps_padded
            )

            scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
            return scores

    def score(self, 
              his_input_titles_padded,
              his_category_emb_padded,
              his_topic_emb_padded,
              his_sentiment_padded,
              his_read_times_padded,
              his_pageviews_padded,
              his_timestamps_padded,
              pred_input_titles_padded,
              pred_category_emb_padded,
              pred_topic_emb_padded,
              pred_sentiment_padded,
              pred_read_times_padded,
              pred_pageviews_padded,
              pred_timestamps_padded,
              impression_timestamps):

        with torch.no_grad():
            user_present = self.userencoder(
                his_input_titles_padded,
                his_category_emb_padded,
                his_topic_emb_padded,
                his_sentiment_padded,
                his_read_times_padded,
                his_pageviews_padded,
                impression_timestamps=impression_timestamps,
                his_timestamps_padded=his_timestamps_padded
            )

            news_present = self.newsencoder(
                pred_input_titles_padded,
                pred_category_emb_padded,
                pred_topic_emb_padded,
                pred_sentiment_padded,
                pred_read_times_padded,
                pred_pageviews_padded,
                impression_timestamps=impression_timestamps,
                pred_timestamps=pred_timestamps_padded
            )

            scores = torch.sum(news_present * user_present, dim=1, keepdim=True)
            return torch.sigmoid(scores)

    def predict(self, 
                his_input_titles_padded,
                his_category_emb_padded,
                his_topic_emb_padded,
                his_sentiment_padded,
                his_read_times_padded,
                his_pageviews_padded,
                his_timestamps_padded,
                pred_input_titles_padded,
                pred_category_emb_padded,
                pred_topic_emb_padded,
                pred_sentiment_padded,
                pred_read_times_padded,
                pred_pageviews_padded,
                pred_timestamps_padded,
                impression_timestamps):
        
        with torch.no_grad():
            scores = self.forward(
                his_input_titles_padded,
                his_category_emb_padded,
                his_topic_emb_padded,
                his_sentiment_padded,
                his_read_times_padded,
                his_pageviews_padded,
                his_timestamps_padded,
                pred_input_titles_padded,
                pred_category_emb_padded,
                pred_topic_emb_padded,
                pred_sentiment_padded,
                pred_read_times_padded,
                pred_pageviews_padded,
                pred_timestamps_padded,
                impression_timestamps
            )
            return torch.sigmoid(scores)

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