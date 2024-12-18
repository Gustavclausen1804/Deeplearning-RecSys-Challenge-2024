import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp

from models_pytorch.layers import AttLayer2, SelfAttention

import torch
import torch.nn as nn
import numpy as np

class NewsEncoderDocVec(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda',
                 use_category=True, use_topic=True, use_numeric=True):
        super(NewsEncoderDocVec, self).__init__()
        self.device = device
        self.use_category = use_category
        self.use_topic = use_topic
        self.use_numeric = use_numeric

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.output_dim = hparams.get('news_output_dim', 200)
        self.units_per_layer = hparams.get('units_per_layer', [512, 512, 512])
        self.docvec_dim = hparams.get('title_size', 30)  # base docvec dimension

        # Embedding dimensions (provided externally)
        self.category_emb_dim = hparams.get("category_emb_dim", 128)
        self.topic_emb_dim = hparams.get("topic_emb_dim", 128)

        # Numeric feature projection parameters
        self.numeric_feature_dim = 3  # sentiment_score, read_time, pageviews
        self.null_indicator_dim = self.numeric_feature_dim  # 1 indicator per numeric feature
        self.numeric_proj_dim = hparams.get('numeric_proj_dim', 16)
        self.dropout_rate = hparams.get('dropout', 0.2)

        # Numeric feature projection
        self.numeric_projection = nn.Sequential(
            nn.BatchNorm1d(self.numeric_feature_dim + self.null_indicator_dim),
            nn.Linear(self.numeric_feature_dim + self.null_indicator_dim, self.numeric_proj_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.numeric_proj_dim),
            nn.Dropout(self.dropout_rate),
            
            # Add layers with reduced dimensionality
            nn.Linear(self.numeric_proj_dim, self.numeric_proj_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.numeric_proj_dim // 2),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.numeric_proj_dim // 2, self.numeric_proj_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(self.numeric_proj_dim // 4),
            nn.Dropout(self.dropout_rate),

            nn.Linear(self.numeric_proj_dim // 4, self.numeric_proj_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.numeric_proj_dim),
            nn.Dropout(self.dropout_rate)
        ).to(device)

        # Calculate combined input dimension dynamically
        input_dim = self.docvec_dim
        if self.use_category:
            input_dim += self.category_emb_dim
        if self.use_topic:
            input_dim += self.topic_emb_dim
        if self.use_numeric:
            input_dim += self.numeric_proj_dim

        # Fully connected layers
        layers = []
        current_dim = input_dim
        for units in self.units_per_layer:
            layers.append(nn.Linear(current_dim, units).to(device))
            layers.append(nn.ReLU().to(device))
            layers.append(nn.BatchNorm1d(units).to(device))
            layers.append(nn.Dropout(self.dropout_rate).to(device))
            current_dim = units

        # Final output layer
        layers.append(nn.Linear(current_dim, self.output_dim).to(device))
        layers.append(nn.ReLU().to(device))

        self.model = nn.Sequential(*layers)

    def forward(self, docvecs, category_emb, topic_emb, sentiment_scores, read_times, pageviews):
        # docvecs: [batch_size, num_titles, docvec_dim]
        # category_emb: [batch_size, num_titles, category_emb_dim] (if use_category)
        # topic_emb: [batch_size, num_titles, topic_emb_dim] (if use_topic)
        # sentiment_scores, read_times, pageviews: [batch_size, num_titles] (if use_numeric)

        docvecs = docvecs.to(self.device)
        batch_size, num_titles, _ = docvecs.shape
        docvecs = docvecs.view(-1, self.docvec_dim)

        combined_features = [docvecs]

        if self.use_category:
            category_emb = category_emb.to(self.device)
            category_emb = category_emb.view(-1, self.category_emb_dim)
            combined_features.append(category_emb)

        if self.use_topic:
            topic_emb = topic_emb.to(self.device)
            topic_emb = topic_emb.view(-1, self.topic_emb_dim)
            combined_features.append(topic_emb)

        if self.use_numeric:
            sentiment_scores = sentiment_scores.to(self.device)
            read_times = read_times.to(self.device)
            pageviews = pageviews.to(self.device)

            numeric_features = torch.cat([
                sentiment_scores.view(-1, 1),
                read_times.view(-1, 1),
                pageviews.view(-1, 1)
            ], dim=-1)

            # Identify null values and create null indicators
            null_indicators = torch.isnan(numeric_features).float()

            # Replace NaN values with 0
            numeric_features[torch.isnan(numeric_features)] = 0

            # Concatenate numeric features with null indicators
            numeric_features_with_nulls = torch.cat([numeric_features, null_indicators], dim=-1)

            # Pass through the numeric projection layer
            numeric_vec = self.numeric_projection(numeric_features_with_nulls)
            combined_features.append(numeric_vec)

        combined = torch.cat(combined_features, dim=-1)  # [batch_size*num_titles, combined_input_dim]
        out = self.model(combined)
        out = out.view(batch_size, num_titles, -1)
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

    def forward(self, his_input_title, his_category_emb, his_topic_emb, his_sentiment_scores, his_read_times, his_pageviews):
        his_input_title = his_input_title.to(self.device)
        his_category_emb = his_category_emb.to(self.device)
        his_topic_emb = his_topic_emb.to(self.device)
        his_sentiment_scores = his_sentiment_scores.to(self.device)
        his_read_times = his_read_times.to(self.device)
        his_pageviews = his_pageviews.to(self.device)

        encoded_titles = self.titleencoder(
            his_input_title,
            his_category_emb,
            his_topic_emb,
            his_sentiment_scores,
            his_read_times,
            his_pageviews
        )  # [batch_size, history_size, output_dim]

        y = self.self_attention(encoded_titles, encoded_titles, encoded_titles)
        y = self.attention_layer(y)
        y = self.user_projection(y)
        return y


class NRMSDocVecModel(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda',
                 use_category=False, use_topic=False, use_numeric=True):
        super(NRMSDocVecModel, self).__init__()
        self.hparams = hparams
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Pass flags to newsencoder
        self.newsencoder = NewsEncoderDocVec(self.hparams, self.seed, self.device,
                                             use_category=use_category,
                                             use_topic=use_topic,
                                             use_numeric=use_numeric)
        self.userencoder = UserEncoderDocVec(self.newsencoder, self.hparams, self.seed, self.device)

        self.to(self.device)
        self.scaler = amp.GradScaler()

    def forward(self,
                his_input_title=None, his_category_emb=None, his_topic_emb=None,
                his_sentiment_scores=None, his_read_times_hist=None, his_pageviews_hist=None,
                pred_input_title=None, pred_category_emb=None, pred_topic_emb=None,
                pred_sentiment_scores=None, pred_read_times=None, pred_pageviews=None):
        
        with torch.cuda.amp.autocast(enabled=False):
            user_present = self.userencoder(
                his_input_title, his_category_emb, his_topic_emb,
                his_sentiment_scores, his_read_times_hist, his_pageviews_hist
            )
            news_present = self.newsencoder(
                pred_input_title,
                pred_category_emb,
                pred_topic_emb,
                pred_sentiment_scores,
                pred_read_times,
                pred_pageviews
            )
            scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
            return scores

    def score(self, 
              his_input_title, his_category_emb, his_topic_emb,
              his_sentiment_scores, his_read_times_hist, his_pageviews_hist,
              pred_input_title_one, pred_category_emb_one, pred_topic_emb_one,
              pred_sentiment_score_one, pred_read_time_one, pred_pageviews_one):
        
        with torch.no_grad():
            user_present = self.userencoder(
                his_input_title, his_category_emb, his_topic_emb,
                his_sentiment_scores, his_read_times_hist, his_pageviews_hist
            )

            news_present_one = self.newsencoder(
                pred_input_title_one.squeeze(1),
                pred_category_emb_one.squeeze(1),
                pred_topic_emb_one.squeeze(1),
                pred_sentiment_score_one.squeeze(1),
                pred_read_time_one.squeeze(1),
                pred_pageviews_one.squeeze(1)
            )

            scores = torch.sum(news_present_one * user_present, dim=1, keepdim=True)
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

    def predict(self, 
                his_input_title, his_category_emb, his_topic_emb,
                his_sentiment_scores, his_read_times_hist, his_pageviews_hist,
                pred_input_title, pred_category_emb, pred_topic_emb,
                pred_sentiment_scores, pred_read_times, pred_pageviews):
        
        with torch.no_grad():
            scores = self.forward(
                his_input_title, his_category_emb, his_topic_emb,
                his_sentiment_scores, his_read_times_hist, his_pageviews_hist,
                pred_input_title, pred_category_emb, pred_topic_emb,
                pred_sentiment_scores, pred_read_times, pred_pageviews
            )
            return torch.sigmoid(scores)
