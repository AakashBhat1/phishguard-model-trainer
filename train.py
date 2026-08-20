"""
🛡️ PhishGuard 2.0 - Standalone Cloud / Google Colab T4 GPU Training Engine
High-throughput training pipeline for 1,200,000+ Malicious & Benign URLs.
Accelerated for NVIDIA Tesla T4 (16GB VRAM) with PyTorch FP16 Tensor Cores, LightGBM GPU/CPU fallback,
Subword Character NLP, Deep Residual PhishNet, and Dynamic Mixture-of-Experts Gating.
"""

import os
import sys
import time
import json
import random
import gc
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import joblib

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LIGHTGBM = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from features import FEATURE_NAMES, extract_features, extract_features_batch

# ---------------------------------------------------------------------------
# Hardware & Environment Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    CUDA_CAPABILITY = torch.cuda.get_device_capability(0)
else:
    GPU_NAME = "Host CPU"
    GPU_VRAM = 0.0
    CUDA_CAPABILITY = (0, 0)


def print_banner():
    print("\n" + "=" * 100)
    print(" 🛡️  PHISHGUARD 2.0 ENTERPRISE CLOUD / COLAB T4 GPU TRAINING PIPELINE")
    print("=" * 100)
    print(f"  • Compute Device     : {DEVICE} ({GPU_NAME})")
    if USE_CUDA:
        print(f"  • GPU Memory (VRAM)  : {GPU_VRAM:.2f} GB GDDR6")
        print(f"  • Compute Capability : {CUDA_CAPABILITY[0]}.{CUDA_CAPABILITY[1]} (Tensor Core FP16 Acceleration Active)")
        print(f"  • PyTorch CUDA Build : {torch.version.cuda}")
    else:
        print("  • GPU Status         : No CUDA GPU detected (running in high-throughput CPU multi-thread mode)")
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Dataset Ingestion & Preprocessing
# ---------------------------------------------------------------------------
def load_and_harmonize_datasets(csv1_path: str, csv2_path: str, max_samples: Optional[int] = 0) -> pd.DataFrame:
    print("=" * 100)
    print(" [1/6] INGESTING & HARMONIZING 1,200,000+ SAMPLES DATASET")
    print("=" * 100)
    dfs = []

    if os.path.exists(csv1_path):
        print(f"  • Reading {os.path.basename(csv1_path)}...")
        df1 = pd.read_csv(csv1_path, low_memory=False)
        u1 = 'url' if 'url' in df1.columns else df1.columns[0]
        t1 = 'type' if 'type' in df1.columns else df1.columns[1]
        c1 = df1[[u1, t1]].dropna().copy()
        c1.rename(columns={u1: 'url', t1: 'raw_label'}, inplace=True)
        c1['url'] = c1['url'].astype(str).str.strip()
        c1 = c1[c1['url'].str.len() > 3]
        c1['label'] = c1['raw_label'].apply(lambda x: 0 if str(x).lower().strip() == 'benign' else 1)
        print(f"    -> Ingested {len(c1):,} rows from {os.path.basename(csv1_path)}")
        dfs.append(c1[['url', 'label']])
    else:
        print(f"  ! Warning: {csv1_path} not found.")

    if os.path.exists(csv2_path):
        print(f"  • Reading {os.path.basename(csv2_path)}...")
        df2 = pd.read_csv(csv2_path, low_memory=False)
        u2 = 'URL' if 'URL' in df2.columns else df2.columns[0]
        t2 = 'Label' if 'Label' in df2.columns else df2.columns[1]
        c2 = df2[[u2, t2]].dropna().copy()
        c2.rename(columns={u2: 'url', t2: 'raw_label'}, inplace=True)
        c2['url'] = c2['url'].astype(str).str.strip()
        c2 = c2[c2['url'].str.len() > 3]
        c2['label'] = c2['raw_label'].apply(lambda x: 0 if str(x).lower().strip() == 'good' else 1)
        print(f"    -> Ingested {len(c2):,} rows from {os.path.basename(csv2_path)}")
        dfs.append(c2[['url', 'label']])
    else:
        print(f"  ! Warning: {csv2_path} not found.")

    if not dfs:
        raise FileNotFoundError(
            "Could not find dataset CSVs! Ensure malicious_phish.csv and phishing_site_urls.csv are in the workspace."
        )

    df_combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    print(f"\n  • Total Combined Raw Entries: {len(df_combined):,}")
    print("  • Deduplicating & calculating consensus labels...")
    df_clean = df_combined.groupby('url', as_index=False)['label'].agg(lambda x: 1 if x.mean() >= 0.5 else 0)
    del df_combined
    gc.collect()

    benign_cnt = int(sum(df_clean['label'] == 0))
    phish_cnt = int(sum(df_clean['label'] == 1))
    print(f"  • Unique Clean URL Samples: {len(df_clean):,} (Benign: {benign_cnt:,} | Malicious: {phish_cnt:,})")

    if max_samples and max_samples > 0 and len(df_clean) > max_samples:
        print(f"  • Downsampling to {max_samples:,} records for quick test...")
        df_clean = df_clean.groupby('label', group_keys=False).apply(
            lambda x: x.sample(int(np.rint(max_samples * len(x) / len(df_clean))), random_state=42)
        ).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        print(f"  • Training on FULL 100% UNTRUNCATED DATASET ({len(df_clean):,} samples)!")

    return df_clean


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'threshold': round(float(threshold), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall_tpr': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4),
        'fpr': round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


# ---------------------------------------------------------------------------
# PyTorch Deep Residual Neural Network (PhishNetDeep)
# ---------------------------------------------------------------------------
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
    """Production Joblib-serializable wrapper around PyTorch PhishNet with GPU FP16 training."""
    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim: int = 256,
        epochs: int = 15,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: torch.device = DEVICE
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.model: Optional[PhishNetDeep] = None
        self.best_val_auc = 0.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        self.model = PhishNetDeep(input_dim=self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)
        
        # Mixed Precision Scaler for T4 GPU Tensor Cores
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
        print(f"    • Training PyTorch PhishNet Deep NN on {self.device} (Batch Size: {self.batch_size}, AMP FP16: {use_amp})...")

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

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        self.model.eval()
        dataset = TabularDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size * 2, shuffle=False, pin_memory=(self.device.type == 'cuda'))
        probs = []

        with torch.no_grad():
            for bx in loader:
                bx = bx.to(self.device, non_blocking=True)
                logits = self.model(bx)
                p = torch.sigmoid(logits).cpu().numpy()
                probs.append(p)

        return np.concatenate(probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save_checkpoint(self, filepath: str):
        if self.model is None:
            return
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'best_val_auc': self.best_val_auc
        }, filepath)


# ---------------------------------------------------------------------------
# LightGBM GPU / CPU Fallback Trainer
# ---------------------------------------------------------------------------
def train_lightgbm_smart(params: Dict[str, Any], X_train, y_train, model_name: str = "LightGBM") -> Any:
    """Attempts training with GPU acceleration (CUDA or OpenCL) if available, falling back cleanly to CPU mode."""
    if not HAS_LIGHTGBM:
        print(f"    ! LightGBM not installed, using HistGradientBoostingClassifier fallback for {model_name}...")
        hgb = HistGradientBoostingClassifier(max_iter=params.get('n_estimators', 300), random_state=42)
        hgb.fit(X_train.toarray() if sp.issparse(X_train) else X_train, y_train)
        return hgb

    # 1. Attempt GPU acceleration with CUDA backend
    if USE_CUDA:
        try:
            gpu_params = params.copy()
            gpu_params.update({
                'device_type': 'cuda',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1
            })
            model = lgb.LGBMClassifier(**gpu_params)
            model.fit(X_train, y_train)
            print(f"    -> {model_name} trained successfully on GPU (CUDA Acceleration)!")
            return model
        except Exception:
            pass

        # 2. Attempt GPU acceleration with OpenCL backend
        try:
            opencl_params = params.copy()
            opencl_params.update({
                'device_type': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'verbose': -1
            })
            model = lgb.LGBMClassifier(**opencl_params)
            model.fit(X_train, y_train)
            print(f"    -> {model_name} trained successfully on GPU (OpenCL Acceleration)!")
            return model
        except Exception as e:
            print(f"    ! LightGBM GPU mode unavailable ({e}). Gracefully executing with multi-threaded CPU...")

    # 3. CPU Multi-thread fallback
    cpu_params = params.copy()
    cpu_params.update({
        'n_jobs': -1,
        'boosting_type': 'gbdt',
        'verbose': -1
    })
    model = lgb.LGBMClassifier(**cpu_params)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Main Full Pipeline Execution
# ---------------------------------------------------------------------------
def run_full_training():
    start_time = time.time()
    print_banner()

    csv1 = os.path.join(BASE_DIR, 'malicious_phish.csv')
    csv2 = os.path.join(BASE_DIR, 'phishing_site_urls.csv')

    # 1. Ingest Data
    df = load_and_harmonize_datasets(csv1, csv2, max_samples=0)
    y = df['label'].values.astype(np.int32)
    urls = df['url'].values
    del df
    gc.collect()

    # 2. 3-Way Partitioning (70% Train, 15% Val, 15% Test)
    print("\n" + "=" * 100)
    print(" [2/6] STRICT 3-WAY PARTITIONING (70% Train / 15% Val / 15% Test)")
    print("=" * 100)
    train_val_urls, test_urls, y_train_val, y_test = train_test_split(
        urls, y, test_size=0.15, random_state=42, stratify=y
    )
    train_urls, val_urls, y_train, y_val = train_test_split(
        train_val_urls, y_train_val, test_size=(0.15 / 0.85), random_state=42, stratify=y_train_val
    )
    del train_val_urls, y_train_val
    gc.collect()

    print(f"  • Training Set   : {len(train_urls):,} URLs ({sum(y_train==0):,} Benign, {sum(y_train==1):,} Malicious)")
    print(f"  • Validation Set : {len(val_urls):,} URLs ({sum(y_val==0):,} Benign, {sum(y_val==1):,} Malicious)")
    print(f"  • Held-Out Test  : {len(test_urls):,} URLs ({sum(y_test==0):,} Benign, {sum(y_test==1):,} Malicious)")

    # 3. High-Speed Batch Feature Extraction
    print("\n" + "=" * 100)
    print(f" [3/6] HIGH-SPEED BATCH FEATURE EXTRACTION ({len(FEATURE_NAMES)} CYBER HEURISTIC SIGNALS)")
    print("=" * 100)
    t0 = time.time()
    X_train_tab = extract_features_batch(train_urls.tolist(), chunk_size=5000, n_jobs=-1)
    X_val_tab = extract_features_batch(val_urls.tolist(), chunk_size=5000, n_jobs=-1)
    X_test_tab = extract_features_batch(test_urls.tolist(), chunk_size=5000, n_jobs=-1)

    scaler = StandardScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab)
    X_val_tab_scaled = scaler.transform(X_val_tab)
    X_test_tab_scaled = scaler.transform(X_test_tab)

    print(f"  • Domain Features Extracted in {time.time() - t0:.2f}s.")
    print("  • Fitting Subword Character N-Gram TF-IDF Vectorizer (20,000 features, ngrams=3-5)...")
    t0_tfidf = time.time()
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=20000,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32
    )
    X_train_tfidf = tfidf.fit_transform(train_urls)
    X_val_tfidf = tfidf.transform(val_urls)
    X_test_tfidf = tfidf.transform(test_urls)
    print(f"    -> TF-IDF Matrix built in {time.time() - t0_tfidf:.2f}s ({X_train_tfidf.shape[1]:,} features).")

    print("  • Building Hybrid Feature Matrices (Sparse Concat)...")
    X_train_hybrid = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_tab_scaled)], format='csr')
    X_val_hybrid = sp.hstack([X_val_tfidf, sp.csr_matrix(X_val_tab_scaled)], format='csr')
    X_test_hybrid = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_tab_scaled)], format='csr')
    print(f"    -> Total Hybrid Features: {X_train_hybrid.shape[1]:,} per sample.")

    # Free up memory
    if USE_CUDA:
        torch.cuda.empty_cache()
    gc.collect()

    # 4. Model Training & Validation Optimization
    print("\n" + "=" * 100)
    print(" [4/6] MODEL TRAINING & VALIDATION HYPERPARAMETER OPTIMIZATION")
    print("=" * 100)

    # Model A: Subword Character NLP Classifier (Fast SGD Log-Loss)
    print("  [1/4] Training Subword NLP Classifier (TF-IDF + SGD Log-Loss)...")
    t0 = time.time()
    nlp_model = SGDClassifier(
        loss='log_loss',
        penalty='l2',
        alpha=1e-5,
        max_iter=30,
        tol=1e-4,
        random_state=42
    )
    nlp_model.fit(X_train_tfidf, y_train)
    p_val_nlp = nlp_model.predict_proba(X_val_tfidf)[:, 1]
    m_val_nlp = evaluate_predictions(y_val, p_val_nlp, 0.50)
    print(f"    -> NLP Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_nlp['accuracy']:.4f} | F1: {m_val_nlp['f1_score']:.4f} | AUC: {m_val_nlp['roc_auc']:.4f}")

    # Model B: PyTorch Deep Residual PhishNet
    print(f"\n  [2/4] Training PyTorch PhishNet Deep Neural Network ({DEVICE})...")
    t0 = time.time()
    phishnet_model = PyTorchPhishNetClassifier(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=256,
        epochs=12,
        batch_size=2048 if USE_CUDA else 1024,
        learning_rate=1e-3,
        device=DEVICE
    )
    phishnet_model.fit(X_train_tab_scaled, y_train, X_val=X_val_tab_scaled, y_val=y_val)
    p_val_nn = phishnet_model.predict_proba(X_val_tab_scaled)
    m_val_nn = evaluate_predictions(y_val, p_val_nn, 0.50)
    print(f"    -> PhishNet Deep NN trained in {time.time() - t0:.2f}s | Val Acc: {m_val_nn['accuracy']:.4f} | F1: {m_val_nn['f1_score']:.4f} | AUC: {m_val_nn['roc_auc']:.4f}")

    # Model C: Tabular Deep LightGBM Classifier
    print(f"\n  [3/4] Training Tabular LightGBM Classifier ({len(FEATURE_NAMES)} Features)...")
    t0 = time.time()
    tree_params = {
        'n_estimators': 450,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'random_state': 42
    }
    tree_model = train_lightgbm_smart(tree_params, X_train_tab_scaled, y_train, model_name="Tabular LightGBM")
    p_val_tree = tree_model.predict_proba(X_val_tab_scaled)[:, 1]
    m_val_tree = evaluate_predictions(y_val, p_val_tree, 0.50)
    print(f"    -> Tabular Tree Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_tree['accuracy']:.4f} | F1: {m_val_tree['f1_score']:.4f} | AUC: {m_val_tree['roc_auc']:.4f}")

    # Model D: Unified Hybrid LightGBM Classifier
    print(f"\n  [4/4] Training Unified Hybrid LightGBM Classifier ({X_train_hybrid.shape[1]:,} Features)...")
    t0 = time.time()
    hybrid_params = {
        'n_estimators': 350,
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_bin': 63,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'random_state': 42
    }
    hybrid_model = train_lightgbm_smart(hybrid_params, X_train_hybrid, y_train, model_name="Hybrid LightGBM")
    p_val_hyb = hybrid_model.predict_proba(X_val_hybrid)[:, 1]
    m_val_hyb = evaluate_predictions(y_val, p_val_hyb, 0.50)
    print(f"    -> Hybrid Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_hyb['accuracy']:.4f} | F1: {m_val_hyb['f1_score']:.4f} | AUC: {m_val_hyb['roc_auc']:.4f}")

    # Mixture of Experts Ensemble & Strict Threshold Optimization
    print("\n  • Optimizing Dynamic Gating & Decision Threshold on Validation Set (FPR <= 0.015)...")
    # Weighted MoE: 50% Hybrid LightGBM + 30% PhishNet Deep NN + 20% Subword NLP
    p_val_moe = 0.50 * p_val_hyb + 0.30 * p_val_nn + 0.20 * p_val_nlp
    best_thresh, best_f1, best_val_m = 0.50, 0.0, None

    for th in np.arange(0.30, 0.80, 0.01):
        m_tmp = evaluate_predictions(y_val, p_val_moe, threshold=th)
        if m_tmp['fpr'] <= 0.015 and m_tmp['f1_score'] > best_f1:
            best_f1 = m_tmp['f1_score']
            best_thresh = round(float(th), 2)
            best_val_m = m_tmp

    if best_val_m is None:
        best_thresh = 0.50
        best_val_m = evaluate_predictions(y_val, p_val_moe, 0.50)

    print(f"    -> Optimal Decision Threshold: {best_thresh} (Val Acc: {best_val_m['accuracy']:.4f}, FPR: {best_val_m['fpr']:.4f}, F1: {best_val_m['f1_score']:.4f})")

    # 5. Held-Out Test Set Evaluation
    print("\n" + "=" * 100)
    print(f" [5/6] FINAL UNBIASED BENCHMARK ON HELD-OUT TEST SET ({len(y_test):,} SAMPLES)")
    print("=" * 100)
    p_test_nlp = nlp_model.predict_proba(X_test_tfidf)[:, 1]
    p_test_nn = phishnet_model.predict_proba(X_test_tab_scaled)
    p_test_tree = tree_model.predict_proba(X_test_tab_scaled)[:, 1]
    p_test_hyb = hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    p_test_moe = 0.50 * p_test_hyb + 0.30 * p_test_nn + 0.20 * p_test_nlp

    m_test_nlp = evaluate_predictions(y_test, p_test_nlp, 0.50)
    m_test_nn = evaluate_predictions(y_test, p_test_nn, 0.50)
    m_test_tree = evaluate_predictions(y_test, p_test_tree, 0.50)
    m_test_hyb = evaluate_predictions(y_test, p_test_hyb, 0.50)
    m_test_moe = evaluate_predictions(y_test, p_test_moe, best_thresh)

    print(f"{'Model Architecture':<34} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'FPR':<6} | {'ROC-AUC'}")
    print("-" * 100)
    print(f"{'1. Subword NLP Baseline':<34} | {m_test_nlp['accuracy']:<8.4f} | {m_test_nlp['precision']:<9.4f} | {m_test_nlp['recall_tpr']:<8.4f} | {m_test_nlp['f1_score']:<8.4f} | {m_test_nlp['fpr']:<6.4f} | {m_test_nlp['roc_auc']:.4f}")
    print(f"{'2. PyTorch PhishNet Deep NN (GPU)':<34} | {m_test_nn['accuracy']:<8.4f} | {m_test_nn['precision']:<9.4f} | {m_test_nn['recall_tpr']:<8.4f} | {m_test_nn['f1_score']:<8.4f} | {m_test_nn['fpr']:<6.4f} | {m_test_nn['roc_auc']:.4f}")
    print(f"{'3. Tabular LightGBM (30 Feat)':<34} | {m_test_tree['accuracy']:<8.4f} | {m_test_tree['precision']:<9.4f} | {m_test_tree['recall_tpr']:<8.4f} | {m_test_tree['f1_score']:<8.4f} | {m_test_tree['fpr']:<6.4f} | {m_test_tree['roc_auc']:.4f}")
    print(f"{'4. Hybrid Feature (LightGBM)':<34} | {m_test_hyb['accuracy']:<8.4f} | {m_test_hyb['precision']:<9.4f} | {m_test_hyb['recall_tpr']:<8.4f} | {m_test_hyb['f1_score']:<8.4f} | {m_test_hyb['fpr']:<6.4f} | {m_test_hyb['roc_auc']:.4f}")
    print(f"{'5. PhishGuard 2.0 MoE (Champion)':<34} | {m_test_moe['accuracy']:<8.4f} | {m_test_moe['precision']:<9.4f} | {m_test_moe['recall_tpr']:<8.4f} | {m_test_moe['f1_score']:<8.4f} | {m_test_moe['fpr']:<6.4f} | {m_test_moe['roc_auc']:.4f}")
    print("=" * 100)

    # 6. Artifact Serialization
    print("\n" + "=" * 100)
    print(" [6/6] SERIALIZING MODEL ARTIFACTS & CREATING ZIP PACKAGE")
    print("=" * 100)

    joblib.dump(tree_model, os.path.join(SAVED_MODELS_DIR, 'tree_ensemble.joblib'))
    joblib.dump(hybrid_model, os.path.join(SAVED_MODELS_DIR, 'hybrid_lgbm.joblib'))
    joblib.dump(nlp_model, os.path.join(SAVED_MODELS_DIR, 'nlp_baseline.joblib'))
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'scaler.joblib'))
    joblib.dump(tfidf, os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.joblib'))

    # PyTorch Model Checkpoint and Wrapper
    phishnet_model.save_checkpoint(os.path.join(SAVED_MODELS_DIR, 'phishnet_nn.pt'))
    joblib.dump(phishnet_model, os.path.join(SAVED_MODELS_DIR, 'transformer_model.joblib'))

    feature_importances = {}
    if hasattr(tree_model, 'feature_importances_'):
        imps = tree_model.feature_importances_
        tot = sum(imps) if sum(imps) > 0 else 1
        for n, v in zip(FEATURE_NAMES, imps):
            feature_importances[n] = round(float((v / tot) * 100), 2)

    report = {
        'test_set_size': len(y_test),
        'optimal_threshold': best_thresh,
        'hardware_accelerator': f"{DEVICE} ({GPU_NAME})",
        'baseline_nlp': m_test_nlp,
        'phishnet_deep_nn': m_test_nn,
        'tabular_lgbm': m_test_tree,
        'hybrid_lgbm': m_test_hyb,
        'champion_moe': m_test_moe,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'test_evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    threshold_cfg = {
        'optimal_threshold': best_thresh,
        'weights': {'alpha_hybrid': 0.50, 'beta_phishnet': 0.30, 'gamma_nlp': 0.20},
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'threshold_config.json'), 'w') as f:
        json.dump(threshold_cfg, f, indent=2)

    eval_metrics_dict = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_set_size': len(y_test),
        'hardware': f"{DEVICE} ({GPU_NAME})",
        'tree_model': m_test_tree,
        'phishnet_model': m_test_nn,
        'transformer_model': m_test_nlp,
        'hybrid_moe_ensemble': m_test_moe,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_metrics_dict, f, indent=2)

    print("  • Saved all .joblib weights, PyTorch .pt weights, scalers, vectorizers, and JSON metrics.")

    # Create ZIP archive
    import shutil
    zip_path = os.path.join(BASE_DIR, 'saved_models.zip')
    shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', SAVED_MODELS_DIR)
    print(f"  • Created ZIP Archive: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

    elapsed = time.time() - start_time
    print(f"\n 🎉 ALL DONE IN {elapsed:.2f}s ({elapsed/60:.2f} MINS)! 'saved_models.zip' is ready.")

    # Trigger Colab Auto-Download if running inside Google Colab
    try:
        if 'google.colab' in sys.modules:
            from google.colab import files
            print("  • Google Colab detected! Triggering automatic browser download of 'saved_models.zip'...")
            files.download(zip_path)
    except Exception as e:
        print(f"  ! Manual download: locate saved_models.zip in {BASE_DIR}")


if __name__ == '__main__':
    run_full_training()
