"""
PhishNet Deep Residual Network - importable module.

WHY THIS FILE EXISTS
--------------------
These classes used to live inside train.py, which Colab runs as __main__. joblib
pickles a class by qualified name, so transformer_model.joblib was written with the
reference "__main__.PyTorchPhishNetClassifier" and could only ever be unpickled by a
process that also happened to be the trainer. Loading it anywhere else - the FastAPI
service included - raises:

    AttributeError: Can't get attribute 'PyTorchPhishNetClassifier' on <module '__main__'>

The service never surfaced the error because models.py loads nlp_baseline.joblib first
and only falls back to transformer_model.joblib, so the PhishNet expert was silently
absent from every production prediction while the reported metrics assumed it present.

Keeping the classes in an importable module makes the pickle reference
"phishnet.PyTorchPhishNetClassifier", which resolves anywhere this file is on the path.
Ship phishnet.py alongside the model artifacts.

Prefer load_phishnet() over joblib when serving: it rebuilds from the plain .pt
checkpoint and carries no pickle compatibility coupling at all.
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.drop(self.act1(self.bn1(self.fc1(x))))
        out = self.act2(self.bn2(self.fc2(out)))
        return out + residual


class PhishNetDeep(nn.Module):
    """Deep Residual MLP Architecture for Tabular and Cyber Heuristic Signals."""

    def __init__(self, input_dim: int = 30, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.middle = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout / 2)
        )
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.input_layer(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.middle(out)
        return self.output_layer(out).squeeze(-1)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class PyTorchPhishNetClassifier:
    """Serializable wrapper around PhishNetDeep with GPU FP16 (AMP) training."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        epochs: int = 15,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device or DEFAULT_DEVICE
        self.model: Optional[PhishNetDeep] = None
        self.best_val_auc = 0.0

    # -- pickle safety -------------------------------------------------------
    # A torch.device does not survive a CPU-only unpickle cleanly, and neither does a
    # CUDA-resident model. Persist weights on the CPU and re-resolve the device on load
    # so an artifact trained on a T4 loads on a CPU-only API host.
    def __getstate__(self):
        state = self.__dict__.copy()
        state['device'] = str(self.device)
        if self.model is not None:
            state['model'] = None
            state['_state_dict'] = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return state

    def __setstate__(self, state):
        sd = state.pop('_state_dict', None)
        self.__dict__.update(state)
        requested = str(state.get('device', 'cpu'))
        if requested.startswith('cuda') and not torch.cuda.is_available():
            requested = 'cpu'
        self.device = torch.device(requested)
        if sd is not None:
            self.model = PhishNetDeep(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
            self.model.load_state_dict(sd)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = None

    # -- training ------------------------------------------------------------
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self.model = PhishNetDeep(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        use_amp = (self.device.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        train_dataset = TabularDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2 if os.name != 'nt' else 0,
            pin_memory=use_amp,
            drop_last=False
        )

        best_weights = None
        print(f"    - Training PhishNet on {self.device} (Batch Size: {self.batch_size}, AMP FP16: {use_amp})...")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            t_start = time.time()

            for bx, by in train_loader:
                bx = bx.to(self.device, non_blocking=True)
                by = by.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.model(bx)
                    loss = criterion(logits, by)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * len(bx)

            scheduler.step()
            avg_loss = total_loss / len(train_dataset)
            epoch_time = time.time() - t_start

            val_auc_str = ""
            if X_val is not None and y_val is not None:
                val_probs = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_probs)
                val_auc_str = f" | Val ROC-AUC: {val_auc:.4f}"
                if val_auc > self.best_val_auc:
                    self.best_val_auc = val_auc
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            print(f"      Epoch [{epoch:02d}/{self.epochs:02d}] - Loss: {avg_loss:.4f}{val_auc_str} ({epoch_time:.2f}s)")

        if best_weights is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})

        return self

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        self.model.eval()
        dataset = TabularDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False,
                            pin_memory=(self.device.type == 'cuda'))
        probs = []
        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.device, non_blocking=True)
                logits = self.model(bx)
                probs.append(torch.sigmoid(logits).float().cpu().numpy())
        return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save_checkpoint(self, filepath: str):
        """Weights-only checkpoint. This, not the joblib pickle, is the durable format."""
        if self.model is None:
            return
        torch.save({
            'state_dict': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'best_val_auc': self.best_val_auc
        }, filepath)


def load_phishnet(checkpoint_path: str, device: Optional[torch.device] = None) -> PyTorchPhishNetClassifier:
    """
    Rebuild a ready-to-serve classifier from phishnet_nn.pt.

    Preferred loading path for the API: no pickle, therefore no class-path or
    library-version coupling to whatever environment produced the artifact.
    """
    device = device or torch.device('cpu')
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    clf = PyTorchPhishNetClassifier(
        input_dim=int(ckpt.get('input_dim', 30)),
        hidden_dim=int(ckpt.get('hidden_dim', 256)),
        device=device
    )
    clf.model = PhishNetDeep(input_dim=clf.input_dim, hidden_dim=clf.hidden_dim)
    clf.model.load_state_dict(ckpt['state_dict'])
    clf.model.to(device)
    clf.model.eval()
    clf.best_val_auc = float(ckpt.get('best_val_auc', 0.0))
    return clf
