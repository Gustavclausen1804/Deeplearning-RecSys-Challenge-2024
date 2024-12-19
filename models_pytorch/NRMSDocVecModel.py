import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp

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
            nn.Dropout(dropout)
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

        combined_input_dim = doc_out_dim
        if self.use_category:
            combined_input_dim += cat_out_dim
        if self.use_topic:
            combined_input_dim += top_out_dim
        if self.use_numeric:
            combined_input_dim += num_out_dim

        final_layers = []
        units_per_layer = hparams.get('units_per_layer', [512, 512, 512])
        input_dim = combined_input_dim
        for units in units_per_layer:
            final_layers.append(nn.Linear(input_dim, units).to(device))
            final_layers.append(nn.ReLU().to(device))
            final_layers.append(nn.BatchNorm1d(units).to(device))
            final_layers.append(nn.Dropout(dropout).to(device))
            input_dim = units

        final_layers.append(nn.Linear(input_dim, self.output_dim).to(device))
        final_layers.append(nn.ReLU().to(device))

        self.final_projection = nn.Sequential(*final_layers)

        # Learnable parameter for discounting based on publication time
        self.use_publication_discount = hparams.get('use_publication_discount', True)
        if self.use_publication_discount:
            self.beta = nn.Parameter(torch.tensor(0.001, dtype=torch.float32, device=device))  # Recommended initial value

    def forward(self, 
                docvecs, category_emb, topic_emb, 
                sentiment_scores, read_times, pageviews, 
                impression_timestamps=None, pred_timestamps=None):
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

        combined_features = [doc_out]
        if cat_out is not None:
            combined_features.append(cat_out)
        if top_out is not None:
            combined_features.append(top_out)
        if num_out is not None:
            combined_features.append(num_out)

        combined = torch.cat(combined_features, dim=-1)
        batch_size, num_titles, _ = docvecs.size()
        out = self.final_projection(combined.view(-1, combined.size(-1)))
        out = out.view(batch_size, num_titles, -1)

        # Publication-time discount
        if self.use_publication_discount and pred_timestamps is not None and impression_timestamps is not None:
            # UNIX seconds difference: delta_news = impression_timestamps - pred_timestamps
            # impression_timestamps: [batch_size], pred_timestamps: [batch_size, num_titles]
            impression_timestamps = impression_timestamps.to(self.device).float().unsqueeze(-1) # [batch_size,1]
            delta_news = impression_timestamps - pred_timestamps.to(self.device).float() # [batch_size, num_titles]
            delta_news = torch.clamp(delta_news, min=0, max=24)  # Clamping to 24 hours

            discount_factors = torch.exp(-self.beta * delta_news)
            discount_factors = discount_factors.unsqueeze(-1) # [batch_size, num_titles, 1]
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

        # Learnable parameter alpha for session-based discount
        self.use_session_discount = hparams.get('use_session_discount', True)
        if self.use_session_discount:
            self.alpha = nn.Parameter(torch.tensor(0.001, dtype=torch.float32, device=device))  # Recommended initial value

    def forward(self, 
                his_input_titles_padded, his_category_emb_padded, his_topic_emb_padded,
                his_sentiment_padded, his_read_times_padded, his_pageviews_padded,
                impression_timestamps=None, his_timestamps_padded=None):

        encoded_titles = self.titleencoder(
            his_input_titles_padded,
            his_category_emb_padded,
            his_topic_emb_padded,
            his_sentiment_padded,
            his_read_times_padded,
            his_pageviews_padded
        )  # [batch_size, history_size, output_dim]

        if self.use_session_discount and his_timestamps_padded is not None and impression_timestamps is not None:
            # delta_user = impression_timestamps - his_timestamps_padded
            impression_timestamps = impression_timestamps.to(self.device).float().unsqueeze(-1) 
            delta_user = impression_timestamps - his_timestamps_padded.to(self.device).float() 
            # print(f"Delta User Min: {delta_user.min().item()}, Max: {delta_user.max().item()}")
            delta_user = torch.clamp(delta_user, min=0, max=48)  # Clamping to 48 hours

            discount_factors = torch.exp(-self.alpha * delta_user) # [batch_size, history_size]
            # print(f"Discount Factors Min: {discount_factors.min().item()}, Max: {discount_factors.max().item()}")

            discount_factors = discount_factors.unsqueeze(-1)      # [batch_size, history_size, 1]
            encoded_titles = encoded_titles * discount_factors

        y = self.self_attention(encoded_titles, encoded_titles, encoded_titles)
        y = self.attention_layer(y)
        y = self.user_projection(y)

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
