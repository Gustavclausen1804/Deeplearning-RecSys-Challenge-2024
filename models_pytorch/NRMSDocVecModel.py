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

        # Set output_dim = head_num * head_dim exactly as in TF
        head_num = hparams["head_num"]
        head_dim = hparams["head_dim"]
        self.output_dim = head_num * head_dim

        self.units_per_layer = hparams.get('units_per_layer', [512, 512, 512])
        input_dim = hparams.get('embedding_dim', 30)

        layers = []
        for units in self.units_per_layer:
            layers.append(nn.Linear(input_dim, units).to(device))
            layers.append(nn.ReLU().to(device))
            layers.append(nn.BatchNorm1d(units).to(device))
            layers.append(nn.Dropout(hparams.get('dropout', 0.2)).to(device))
            input_dim = units

        # Final layer matches TF dimension
        layers.append(nn.Linear(input_dim, self.output_dim).to(device))
        layers.append(nn.ReLU().to(device))

        self.model = nn.Sequential(*layers)


    def forward(self, x):
        x = x.to(self.device)

        if len(x.shape) == 3:  # Handle batched input with multiple titles
            batch_size, num_titles, docvec_dim = x.shape
            x = x.view(-1, docvec_dim)  # Flatten to process all titles together
        else:
            batch_size = None
            num_titles = None

        out = self.model(x)

        if batch_size is not None and num_titles is not None:
            out = out.view(batch_size, num_titles, -1)

        return out

class UserEncoderDocVec(nn.Module):
    def __init__(self, titleencoder, hparams, seed, device='cuda'):
        super(UserEncoderDocVec, self).__init__()
        self.device = device
        self.titleencoder = titleencoder
        self.self_attention = SelfAttention(
            multiheads=hparams["head_num"], 
            head_dim=hparams["head_dim"], 
            seed=seed,
            device=device
        )
        self.attention_layer = AttLayer2(
            dim=hparams["attention_hidden_dim"],
            seed=seed,
            device=device
        ).to(device)

        # Remove the user_projection layer entirely
        # The TF version returns the output of AttLayer2 directly as user embedding.
        # Ensure that hparams["attention_hidden_dim"] == hparams["head_num"] * hparams["head_dim"]
        # or adjust as needed based on your TF code specifics.

    def forward(self, his_input_title):
        his_input_title = his_input_title.to(self.device)
        batch_size, history_size, docvec_dim = his_input_title.size()
        encoded_titles = self.titleencoder(his_input_title)
        click_title_presents = encoded_titles

        y = self.self_attention([click_title_presents, click_title_presents, click_title_presents])
        y = self.attention_layer(y)

        return y


class NRMSDocVecModel(nn.Module):
    def __init__(self, hparams, seed=42, device='cuda'):
        super().__init__()
        self.hparams = hparams
        self.seed = seed
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Define NewsEncoder and UserEncoder with document vector adaptation
        self.newsencoder = self._build_newsencoder()
        self.userencoder = self._build_userencoder(self.newsencoder)

        # Move model to device
        self.to(self.device)
        self.scaler = amp.GradScaler()

    def _build_newsencoder(self):
        return NewsEncoderDocVec(self.hparams, self.seed, self.device)

    def _build_userencoder(self, titleencoder):
        return UserEncoderDocVec(titleencoder, self.hparams, self.seed, self.device)

    def forward(self, his_input_title, pred_input_title):
        with torch.cuda.amp.autocast(enabled=False):
            #print(f"NRMSDocVecModel - Input his_input_title shape: {his_input_title.shape}")
            #print(f"NRMSDocVecModel - Input pred_input_title shape: {pred_input_title.shape}")
            
            his_input_title = his_input_title.to(self.device)
            pred_input_title = pred_input_title.to(self.device)

            user_present = self.userencoder(his_input_title)
            #print(f"NRMSDocVecModel - User presentation shape: {user_present.shape}")

            news_present = self.newsencoder(pred_input_title)
            #print(f"NRMSDocVecModel - News presentation shape: {news_present.shape}")

            # Ensure proper dimensions for batch matrix multiplication
            if len(news_present.shape) == 2:  # If flattened, reshape to 3D
                batch_size = his_input_title.size(0)
                num_titles = pred_input_title.size(1)
                news_present = news_present.view(batch_size, num_titles, -1)

            #print(f"NRMSDocVecModel - Reshaped news presentation shape: {news_present.shape}")

            # Compute scores
            try:
                scores = torch.bmm(news_present, user_present.unsqueeze(-1)).squeeze(-1)
                #print(f"NRMSDocVecModel - Scores shape: {scores.shape}")
            except RuntimeError as e:
                #print(f"Error during batch matrix multiplication: {e}")
                #print(f"news_present shape: {news_present.shape}, user_present shape: {user_present.unsqueeze(-1).shape}")
                raise

            return scores

    def score(self, his_input_title, pred_input_title_one):
        #print(f"NRMSDocVecModel - Scoring his_input_title shape: {his_input_title.shape}")
        #print(f"NRMSDocVecModel - Scoring pred_input_title_one shape: {pred_input_title_one.shape}")
        his_input_title = his_input_title.to(self.device)
        pred_input_title_one = pred_input_title_one.to(self.device)

        user_present = self.userencoder(his_input_title)
        #print(f"NRMSDocVecModel - User presentation for scoring: {user_present.shape}")

        news_present_one = self.newsencoder(pred_input_title_one.squeeze(1))
        #print(f"NRMSDocVecModel - News presentation for scoring: {news_present_one.shape}")

        scores = torch.sum(news_present_one * user_present, dim=1, keepdim=True)
        #print(f"NRMSDocVecModel - Final scores: {scores.shape}")
        return torch.sigmoid(scores)

    def get_loss(self, criterion="cross_entropy"):
        #print(f"NRMSDocVecModel - Loss function: {criterion}")
        if criterion == "cross_entropy":
            return nn.CrossEntropyLoss(ignore_index=-1).to(self.device)
        elif criterion == "log_loss":
            return nn.BCELoss().to(self.device)
        else:
            raise ValueError(f"Loss function not defined: {criterion}")

    def get_optimizer(self, hparams_nrms):
        if hparams_nrms.__dict__['optimizer'] == "adam":
            return optim.Adam(self.parameters(), lr=hparams_nrms.__dict__['learning_rate'], weight_decay=hparams_nrms.__dict__['weight_decay'])
        else:
            raise ValueError(f"Optimizer not defined")

    def predict(self, his_input_title, pred_input_title):
        with torch.no_grad():
            scores = self.forward(his_input_title, pred_input_title)        
            return torch.sigmoid(scores)
