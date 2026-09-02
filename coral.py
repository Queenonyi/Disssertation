"""Deep CORAL implementation."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def coral_loss(source, target):
    d = source.size(1)
    source_c = source - source.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)
    cov_s = (source_c.t() @ source_c) / (source_c.size(0) - 1)
    cov_t = (target_c.t() @ target_c) / (target_c.size(0) - 1)
    loss = torch.sum((cov_s - cov_t) ** 2) / (4 * d * d)
    return loss

class CORALNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        self.feature = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, 1)
    
    def forward(self, x):
        feat = self.feature(x)
        return self.classifier(feat), feat  # Logits, not sigmoid

class CORALTrainer:
    def __init__(self, input_dim, alpha=1.0, lr=1e-3, epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CORALNet(input_dim).to(self.device)
        self.alpha = alpha
        self.epochs = epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []
    
    def fit(self, X_source, y_source, X_target):
        X_s = torch.tensor(np.array(X_source, dtype=np.float32), dtype=torch.float32).to(self.device)
        y_s = torch.tensor(np.array(y_source, dtype=np.float32), dtype=torch.float32).unsqueeze(1).to(self.device)
        X_t = torch.tensor(np.array(X_target, dtype=np.float32), dtype=torch.float32).to(self.device)
        
        for epoch in range(self.epochs):
            self.model.train()
            pred_s, feat_s = self.model(X_s)
            label_loss = self.criterion(pred_s, y_s)
            
            _, feat_t = self.model(X_t)
            coral = coral_loss(feat_s, feat_t)
            total_loss = label_loss + self.alpha * coral
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Label Loss={label_loss.item():.4f}, CORAL={coral.item():.4f}")
            
            self.history.append({
                "epoch": epoch,
                "label_loss": label_loss.item(),
                "coral_loss": coral.item(),
                "total_loss": total_loss.item(),
            })
        return self
    
    def get_history(self):
        return self.history
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float32).to(self.device)
            logits, _ = self.model(X_t)
            prob = torch.sigmoid(logits).cpu().numpy()
            return prob
    
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)