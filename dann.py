"""Domain-Adversarial Neural Network (DANN) implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)  # No sigmoid --- BCEWithLogitsLoss handles it

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)  # No sigmoid --- BCEWithLogitsLoss handles it

class DANNTrainer:
    def __init__(self, input_dim, lambda_init=0.0, lambda_max=1.0, lr=1e-3, epochs=100, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = FeatureExtractor(input_dim).to(self.device)
        self.label_predictor = LabelPredictor(32).to(self.device)
        self.domain_discriminator = DomainDiscriminator(32).to(self.device)
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) +
            list(self.label_predictor.parameters()) +
            list(self.domain_discriminator.parameters()),
            lr=lr
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []
    
    def _get_lambda(self, epoch):
        p = float(epoch) / self.epochs
        return self.lambda_max * (2. / (1. + np.exp(-10 * p)) - 1)
    
    def fit(self, X_source, y_source, X_target):
        X_s = torch.tensor(np.array(X_source, dtype=np.float32), dtype=torch.float32).to(self.device)
        y_s = torch.tensor(np.array(y_source, dtype=np.float32), dtype=torch.float32).unsqueeze(1).to(self.device)
        X_t = torch.tensor(np.array(X_target, dtype=np.float32), dtype=torch.float32).to(self.device)
        
        for epoch in range(self.epochs):
            lambda_ = self._get_lambda(epoch)
            self.feature_extractor.train()
            self.label_predictor.train()
            self.domain_discriminator.train()
            
            feat_s = self.feature_extractor(X_s)
            pred_s = self.label_predictor(feat_s)
            label_loss = self.criterion(pred_s, y_s)
            
            domain_s = torch.ones(feat_s.size(0), 1).to(self.device)
            domain_t = torch.zeros(X_t.size(0), 1).to(self.device)
            
            rev_feat_s = GradientReversalLayer.apply(feat_s, lambda_)
            domain_pred_s = self.domain_discriminator(rev_feat_s)
            domain_loss_s = self.criterion(domain_pred_s, domain_s)
            
            feat_t = self.feature_extractor(X_t)
            rev_feat_t = GradientReversalLayer.apply(feat_t, lambda_)
            domain_pred_t = self.domain_discriminator(rev_feat_t)
            domain_loss_t = self.criterion(domain_pred_t, domain_t)
            
            domain_loss = domain_loss_s + domain_loss_t
            total_loss = label_loss + domain_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.feature_extractor.parameters()) +
                list(self.label_predictor.parameters()) +
                list(self.domain_discriminator.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Label Loss={label_loss.item():.4f}, Domain Loss={domain_loss.item():.4f}")
            
            self.history.append({
                "epoch": epoch,
                "label_loss": label_loss.item(),
                "domain_loss": domain_loss.item(),
                "total_loss": total_loss.item(),
            })
        return self
    
    def get_history(self):
        return self.history
    
    def predict_proba(self, X):
        self.feature_extractor.eval()
        self.label_predictor.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32).to(self.device)
            feat = self.feature_extractor(X_t)
            logits = self.label_predictor(feat)
            prob = torch.sigmoid(logits).cpu().numpy()
            return prob
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)