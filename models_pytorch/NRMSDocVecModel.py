import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.cuda.amp as amp

from models_pytorch.layers import AttLayer2, SelfAttention

class NewsEncoderDocVec(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super(NewsEncoderDocVec, self).__init__()
        self.device = device
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
        self.numeric_proj_dim = hparams.get('numeric_proj_dim', 16)
        self.numeric_projection = nn.Linear(self.numeric_feature_dim, self.numeric_proj_dim).to(device)

        # Input dimension after concatenation:
        # docvec_dim + category_emb_dim + topic_emb_dim + numeric_proj_dim
        combined_input_dim = self.docvec_dim + self.category_emb_dim + self.topic_emb_dim + self.numeric_proj_dim

        layers = []
        input_dim = combined_input_dim
        for units in self.units_per_layer:
            layers.append(nn.Linear(input_dim, units).to(device))
            layers.append(nn.ReLU().to(device))
            layers.append(nn.BatchNorm1d(units).to(device))
            layers.append(nn.Dropout(hparams.get('dropout', 0.2)).to(device))
            input_dim = units

        # Final output layer
        layers.append(nn.Linear(input_dim, self.output_dim).to(device))
        layers.append(nn.ReLU().to(device))

        self.model = nn.Sequential(*layers)

    def forward(self, docvecs, category_emb, topic_emb, sentiment_scores, read_times, pageviews):
        # docvecs: [batch_size, num_titles, docvec_dim]
        # category_emb: [batch_size, num_titles, category_emb_dim]
        # topic_emb: [batch_size, num_titles, topic_emb_dim]
        # sentiment_scores, read_times, pageviews: [batch_size, num_titles]

        docvecs = docvecs.to(self.device)
        category_emb = category_emb.to(self.device)
        topic_emb = topic_emb.to(self.device)
        sentiment_scores = sentiment_scores.to(self.device)
        read_times = read_times.to(self.device)
        pageviews = pageviews.to(self.device)

        batch_size, num_titles, _ = docvecs.shape

        # Flatten
        docvecs = docvecs.view(-1, self.docvec_dim)
        category_emb = category_emb.view(-1, self.category_emb_dim)
        topic_emb = topic_emb.view(-1, self.topic_emb_dim)

        numeric_features = torch.cat([
            sentiment_scores.view(-1, 1),
            read_times.view(-1, 1),
            pageviews.view(-1, 1)
        ], dim=-1)  # [batch_size*num_titles, 3]

        numeric_vec = self.numeric_projection(numeric_features) # [batch_size*num_titles, numeric_proj_dim]

        combined = torch.cat([docvecs, category_emb, topic_emb, numeric_vec], dim=-1)
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
        # his_input_title: [batch_size, history_size, docvec_dim]
        # his_category_emb: [batch_size, history_size, category_emb_dim]
        # his_topic_emb: [batch_size, history_size, topic_emb_dim]
        # his_sentiment_scores, his_read_times, his_pageviews: [batch_size, history_size]

        his_input_title = his_input_title.to(self.device)
        his_category_emb = his_category_emb.to(self.device)
        his_topic_emb = his_topic_emb.to(self.device)
        his_sentiment_scores = his_sentiment_scores.to(self.device)
        his_read_times = his_read_times.to(self.device)
        his_pageviews = his_pageviews.to(self.device)

        batch_size, history_size, docvec_dim = his_input_title.size()

        # Encode user history articles using the same encoder as candidate articles
        # Flatten them first if needed
        encoded_titles = self.titleencoder(
            his_input_title,
            his_category_emb,
            his_topic_emb,
            his_sentiment_scores,
            his_read_times,
            his_pageviews
        )  # [batch_size, history_size, output_dim]

        # Apply self-attention over user history embeddings
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

        # Build encoders
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(self.newsencoder)

        self.to(self.device)
        self.scaler = amp.GradScaler()

    def _build_newsencoder(self):
        return NewsEncoderDocVec(self.hparams, self.seed, self.device)

    def _build_userencoder(self, titleencoder):
        return UserEncoderDocVec(titleencoder, self.hparams, self.seed, self.device)

    def forward(self,
                his_input_title=None, his_category_emb=None, his_topic_emb=None,
                his_sentiment_scores=None, his_read_times_hist=None, his_pageviews_hist=None,
                pred_input_title=None, pred_category_emb=None, pred_topic_emb=None,
                pred_sentiment_scores=None, pred_read_times=None, pred_pageviews=None):
        
        with torch.cuda.amp.autocast(enabled=False):
            # Encode user representation with all features
            user_present = self.userencoder(
                his_input_title, his_category_emb, his_topic_emb,
                his_sentiment_scores, his_read_times_hist, his_pageviews_hist
            )

            # Encode candidate news representation with all features
            news_present = self.newsencoder(
                pred_input_title,
                pred_category_emb,
                pred_topic_emb,
                pred_sentiment_scores,
                pred_read_times,
                pred_pageviews
            )

            # Compute scores
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
