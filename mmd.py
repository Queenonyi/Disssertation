"""Maximum Mean Discrepancy (MMD) regularization --- memory-safe."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def gaussian_kernel(x, y, sigma=1.0):
    with torch.no_grad():
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2))

def compute_mmd(x, y, kernel=gaussian_kernel):
    """Compute MMD^2 between x and y."""
    x_kernel = kernel(x, x)
    y_kernel = kernel(y, y)
    xy_kernel = kernel(x, y)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

class MMDNet(nn.Module):
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
        return self.classifier(feat), feat

class MMDTrainer:
    def __init__(self, input_dim, beta=1.0, sigma=1.0, lr=1e-3, epochs=30, mmd_sample_size=2000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MMDNet(input_dim).to(self.device)
        self.beta = beta
        self.sigma = sigma
        self.epochs = epochs
        self.mmd_sample_size = mmd_sample_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()
        self.history = []
    
    def fit(self, X_source, y_source, X_target):
        X_s = torch.tensor(np.array(X_source, dtype=np.float32), dtype=torch.float32).to(self.device)
        y_s = torch.tensor(np.array(y_source, dtype=np.float32), dtype=torch.float32).unsqueeze(1).to(self.device)
        X_t_full = torch.tensor(np.array(X_target, dtype=np.float32), dtype=torch.float32).to(self.device)
        
        for epoch in range(self.epochs):
            if len(X_t_full) > self.mmd_sample_size:
                idx = torch.randperm(len(X_t_full))[:self.mmd_sample_size]
                X_t = X_t_full[idx]
            else:
                X_t = X_t_full
            
            self.model.train()
            pred_s, feat_s = self.model(X_s)
            label_loss = self.criterion(pred_s, y_s)
            
            _, feat_t = self.model(X_t)
            mmd = compute_mmd(feat_s, feat_t)
            total_loss = label_loss + self.beta * mmd
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Label Loss={label_loss.item():.4f}, MMD={mmd.item():.4f}")
            
            self.history.append({
                "epoch": epoch,
                "label_loss": label_loss.item(),
                "mmd_loss": mmd.item(),
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