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

import sklearn
from sklearn.model_selection import train_test_split, GroupShuffleSplit
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

from features import (
    FEATURE_NAMES, extract_features, extract_features_batch,
    normalize_url, registered_domain, NORMALIZATION_VERSION
)
# PhishNet lives in its own module so the pickled classifier resolves as
# phishnet.PyTorchPhishNetClassifier instead of __main__.PyTorchPhishNetClassifier.
# See the module docstring in phishnet.py for why that mattered.
from phishnet import PyTorchPhishNetClassifier, PhishNetDeep, load_phishnet

# ---------------------------------------------------------------------------
# Hardware & Environment Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Fraction of benign training hosts injected back as bare-domain rows. In the raw
# corpus 0.00% of benign URLs are bare domains against 7.69% of malicious ones, so a
# bare domain like "sg.com" is territory the model has never seen labelled benign and
# it defaults to p ~ 1.0. Set to 0.0 to disable the augmentation.
BARE_DOMAIN_AUGMENT_FRACTION = 0.5

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
MASTER_PARQUET = os.path.join(BASE_DIR, 'phishguard_master_dataset.parquet')
MASTER_CSV = os.path.join(BASE_DIR, 'phishguard_master_dataset.csv')
MASTER_MANIFEST = os.path.join(BASE_DIR, 'master_dataset_manifest.json')


def load_master_dataset(max_samples: Optional[int] = 0) -> Optional[pd.DataFrame]:
    """
    Load the consolidated master dataset if consolidate_datasets.py has produced one.

    The master file is already normalised, cross-corpus deduplicated, consensus-labelled
    and group-tagged, so this path skips straight to sampling. Parquet is preferred: it
    is ~45 MB against ~80 MB and loads in a fraction of the time, which matters on a
    Colab runtime that re-uploads the file every session.

    Returns None when no master file exists, so the caller falls back to the raw CSVs.
    """
    path = MASTER_PARQUET if os.path.exists(MASTER_PARQUET) else (
        MASTER_CSV if os.path.exists(MASTER_CSV) else None)
    if path is None:
        return None

    print(f"  • Master dataset found: {os.path.basename(path)}")
    try:
        if path.endswith('.parquet'):
            df = pd.read_parquet(path, columns=['url', 'label', 'group'])
        else:
            df = pd.read_csv(path, usecols=['url', 'label', 'group'], low_memory=False)
    except ImportError as e:
        # Parquet present but no engine installed on this runtime.
        if not os.path.exists(MASTER_CSV):
            print(f"  ! Cannot read {os.path.basename(path)} ({e}) and no CSV fallback exists.")
            return None
        print(f"  ! Parquet engine missing ({e}); falling back to the CSV copy.")
        df = pd.read_csv(MASTER_CSV, usecols=['url', 'label', 'group'], low_memory=False)

    df = df.dropna(subset=['url', 'label', 'group'])
    df['label'] = df['label'].astype(int)

    # The master file records which normalisation and which conflict rule built it.
    # Training against a master built by a different features.py is the same silent
    # skew that the serving manifest check guards against, so refuse to be quiet here.
    if os.path.exists(MASTER_MANIFEST):
        try:
            with open(MASTER_MANIFEST, 'r', encoding='utf-8') as f:
                mm = json.load(f)
            built_with = mm.get('url_normalization')
            if built_with and built_with != NORMALIZATION_VERSION:
                print(f"  ! WARNING: master dataset was built with normalisation '{built_with}' "
                      f"but features.py implements '{NORMALIZATION_VERSION}'. "
                      f"Re-run consolidate_datasets.py before trusting this run.")
            audit = mm.get('surface_form_audit', {}).get('master', {})
            if audit:
                print(f"    -> surface-form path-rule accuracy: {audit.get('path_rule_accuracy')} "
                      f"(0.50 = no formatting shortcut)")
            conflicts = mm.get('totals', {}).get('label_conflicts')
            if conflicts:
                print(f"    -> {conflicts:,} cross-corpus label conflicts, policy: "
                      f"{mm.get('conflict_policy')}")
        except Exception as e:
            print(f"  ! Could not read master manifest: {e}")

    benign_cnt = int((df['label'] == 0).sum())
    phish_cnt = int((df['label'] == 1).sum())
    print(f"  • Loaded {len(df):,} pre-cleaned URLs "
          f"(Benign: {benign_cnt:,} | Malicious: {phish_cnt:,}) across "
          f"{df['group'].nunique():,} registered domains")

    if max_samples and max_samples > 0 and len(df) > max_samples:
        print(f"  • Downsampling to {max_samples:,} records for quick test...")
        df = (df.groupby('label', group_keys=False, sort=False)
              .sample(frac=max_samples / len(df), random_state=42)
              .sample(frac=1.0, random_state=42)
              .reset_index(drop=True))
    else:
        print(f"  • Training on FULL 100% MASTER DATASET ({len(df):,} samples)!")

    return df.reset_index(drop=True)


def load_and_harmonize_datasets(csv1_path: str, csv2_path: str, max_samples: Optional[int] = 0) -> pd.DataFrame:
    print("=" * 100)
    print(" [1/6] INGESTING & HARMONIZING DATASET")
    print("=" * 100)

    # Prefer the consolidated master dataset; fall back to harmonising the raw CSVs so
    # the trainer still runs on a checkout where consolidate_datasets.py was never run.
    master = load_master_dataset(max_samples=max_samples)
    if master is not None:
        return master
    print("  • No master dataset present — harmonising raw CSVs "
          "(run consolidate_datasets.py to skip this step).")

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

    # Normalise BEFORE deduplication. The two corpora disagree on surface form
    # (43.6% of malicious rows carry a scheme against 9.3% of benign ones), and a
    # char_wb TF-IDF fitted on the raw strings learns that difference instead of
    # learning the URL. Normalising first also collapses http/https duplicates of the
    # same target into one row.
    print(f"  • Normalising URLs ({NORMALIZATION_VERSION})...")
    df_combined['url'] = df_combined['url'].map(normalize_url)
    df_combined = df_combined[df_combined['url'].str.len() > 3]

    print("  • Deduplicating & calculating consensus labels...")
    df_clean = df_combined.groupby('url', as_index=False)['label'].mean()
    df_clean['label'] = (df_clean['label'] >= 0.5).astype(int)
    del df_combined
    gc.collect()

    # Grouping key for the split. Without it the same host lands in train and test.
    print("  • Deriving registered-domain grouping keys...")
    df_clean['group'] = df_clean['url'].map(registered_domain)
    print(f"    -> {df_clean['group'].nunique():,} unique registered domains across "
          f"{len(df_clean):,} URLs ({len(df_clean) / max(1, df_clean['group'].nunique()):.1f} URLs per domain)")

    benign_cnt = int(sum(df_clean['label'] == 0))
    phish_cnt = int(sum(df_clean['label'] == 1))
    print(f"  • Unique Clean URL Samples: {len(df_clean):,} (Benign: {benign_cnt:,} | Malicious: {phish_cnt:,})")

    if max_samples and max_samples > 0 and len(df_clean) > max_samples:
        print(f"  • Downsampling to {max_samples:,} records for quick test...")
        frac = max_samples / len(df_clean)
        keep = df_clean.groupby('label', group_keys=False, sort=False).sample(
            frac=frac, random_state=42
        ).index
        df_clean = df_clean.loc[keep].sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        print(f"  • Training on FULL 100% UNTRUNCATED DATASET ({len(df_clean):,} samples)!")

    return df_clean


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    # labels=[0, 1] keeps ravel() at four cells. Without it a slice that happens to be
    # single-class returns a 1x1 matrix and the unpack raises ValueError.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'threshold': round(float(threshold), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall_tpr': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.0,
        'fpr': round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


# ---------------------------------------------------------------------------
# PyTorch Deep Residual Neural Network (PhishNetDeep)
# ---------------------------------------------------------------------------
# ResidualBlock / PhishNetDeep / TabularDataset / PyTorchPhishNetClassifier now live
# in phishnet.py and are imported at the top of this file. They were moved out because
# joblib pickles a class by qualified name: defined here they serialised as
# "__main__.PyTorchPhishNetClassifier", so transformer_model.joblib could not be
# unpickled by anything except the trainer process itself.


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
def run_full_training(max_samples: int = 0):
    start_time = time.time()
    print_banner()

    csv1 = os.path.join(BASE_DIR, 'malicious_phish.csv')
    csv2 = os.path.join(BASE_DIR, 'phishing_site_urls.csv')

    # 1. Ingest Data
    df = load_and_harmonize_datasets(csv1, csv2, max_samples=max_samples)

    # 2. 3-Way Partitioning grouped by registered domain (70% / 15% / 15%)
    print("\n" + "=" * 100)
    print(" [2/6] DOMAIN-GROUPED 3-WAY PARTITIONING (70% Train / 15% Val / 15% Test)")
    print("=" * 100)
    # A stratified *row* split leaks: the corpus averages ~3 URLs per registered
    # domain, so most test hosts also appear in train and the model is scored on
    # domains it has memorised. GroupShuffleSplit keeps every domain wholly inside one
    # partition. Expect the headline AUC to fall below the old ~0.998 - that fall is
    # the measurement error being removed, not a regression.
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    idx_train_val, idx_test = next(gss_test.split(df, df['label'], groups=df['group']))
    df_train_val = df.iloc[idx_train_val].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)

    gss_val = GroupShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=42)
    idx_train, idx_val = next(gss_val.split(df_train_val, df_train_val['label'], groups=df_train_val['group']))
    df_train = df_train_val.iloc[idx_train].reset_index(drop=True)
    df_val = df_train_val.iloc[idx_val].reset_index(drop=True)
    del df, df_train_val
    gc.collect()

    # Bare-domain benign augmentation, TRAIN SPLIT ONLY so it cannot touch the score.
    # Injecting the registered domains of benign training URLs as standalone benign
    # rows teaches the model that a short bare domain is an ordinary benign shape.
    if BARE_DOMAIN_AUGMENT_FRACTION > 0:
        benign_hosts = (
            df_train.loc[df_train['label'] == 0, 'group']
            .drop_duplicates()
            .sample(frac=BARE_DOMAIN_AUGMENT_FRACTION, random_state=7)
        )
        aug = pd.DataFrame({'url': benign_hosts.values, 'label': 0, 'group': benign_hosts.values})
        before = len(df_train)
        df_train = (
            pd.concat([df_train, aug], ignore_index=True)
            .drop_duplicates(subset='url', keep='first')
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        print(f"  • Bare-domain augmentation: +{len(df_train) - before:,} benign root-domain rows")

    train_urls, y_train = df_train['url'].values, df_train['label'].values.astype(np.int32)
    val_urls, y_val = df_val['url'].values, df_val['label'].values.astype(np.int32)
    test_urls, y_test = df_test['url'].values, df_test['label'].values.astype(np.int32)

    leaked = len(set(df_train['group']) & set(df_test['group']))
    print(f"  • Training Set   : {len(train_urls):,} URLs ({sum(y_train==0):,} Benign, {sum(y_train==1):,} Malicious)")
    print(f"  • Validation Set : {len(val_urls):,} URLs ({sum(y_val==0):,} Benign, {sum(y_val==1):,} Malicious)")
    print(f"  • Held-Out Test  : {len(test_urls):,} URLs ({sum(y_test==0):,} Benign, {sum(y_test==1):,} Malicious)")
    print(f"  • Domain overlap between train and test: {leaked} (must be 0)")
    del df_train, df_val, df_test
    gc.collect()

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
        # LightGBM ignores subsample entirely unless subsample_freq > 0, so the
        # bagging these configs asked for was never actually happening.
        'subsample_freq': 1,
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
        'subsample_freq': 1,
        'colsample_bytree': 0.80,
        'random_state': 42
    }
    hybrid_model = train_lightgbm_smart(hybrid_params, X_train_hybrid, y_train, model_name="Hybrid LightGBM")
    p_val_hyb = hybrid_model.predict_proba(X_val_hybrid)[:, 1]
    m_val_hyb = evaluate_predictions(y_val, p_val_hyb, 0.50)
    print(f"    -> Hybrid Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_hyb['accuracy']:.4f} | F1: {m_val_hyb['f1_score']:.4f} | AUC: {m_val_hyb['roc_auc']:.4f}")

    # Mixture of Experts Ensemble & Strict Threshold Optimization
    def tune_threshold(p_val, max_fpr=0.015):
        """Highest-F1 threshold subject to an FPR ceiling, falling back to 0.50."""
        best_t, best_f1_local, best_m = 0.50, 0.0, None
        for th in np.arange(0.30, 0.80, 0.01):
            m_tmp = evaluate_predictions(y_val, p_val, threshold=th)
            if m_tmp['fpr'] <= max_fpr and m_tmp['f1_score'] > best_f1_local:
                best_f1_local = m_tmp['f1_score']
                best_t = round(float(th), 2)
                best_m = m_tmp
        if best_m is None:
            best_t = 0.50
            best_m = evaluate_predictions(y_val, p_val, 0.50)
        return best_t, best_m

    print("\n  • Optimizing Dynamic Gating & Decision Threshold on Validation Set (FPR <= 0.015)...")
    # Weighted MoE: 50% Hybrid LightGBM + 30% PhishNet Deep NN + 20% Subword NLP
    p_val_moe = 0.50 * p_val_hyb + 0.30 * p_val_nn + 0.20 * p_val_nlp
    best_thresh, best_val_m = tune_threshold(p_val_moe)
    print(f"    -> 3-expert MoE threshold: {best_thresh} (Val Acc: {best_val_m['accuracy']:.4f}, "
          f"FPR: {best_val_m['fpr']:.4f}, F1: {best_val_m['f1_score']:.4f})")

    # A threshold is only valid for the exact score it was tuned against. ml_service
    # fuses two experts with instance-level gating weights, alpha*p_nlp + beta*p_hybrid,
    # which is NOT the 3-expert blend above - so shipping only the MoE threshold left
    # the server applying a cut-off tuned for a score it never computes. Tune and ship
    # a second threshold for the serving fusion at its default gate (alpha=0.45,
    # beta=0.55) so the two agree out of the box.
    SERVING_ALPHA, SERVING_BETA = 0.45, 0.55
    p_val_serving = SERVING_ALPHA * p_val_nlp + SERVING_BETA * p_val_hyb
    serving_thresh, serving_val_m = tune_threshold(p_val_serving)
    print(f"    -> serving fusion threshold (alpha={SERVING_ALPHA}, beta={SERVING_BETA}): {serving_thresh} "
          f"(Val Acc: {serving_val_m['accuracy']:.4f}, FPR: {serving_val_m['fpr']:.4f}, F1: {serving_val_m['f1_score']:.4f})")

    # 5. Held-Out Test Set Evaluation
    print("\n" + "=" * 100)
    print(f" [5/6] FINAL UNBIASED BENCHMARK ON HELD-OUT TEST SET ({len(y_test):,} SAMPLES)")
    print("=" * 100)
    p_test_nlp = nlp_model.predict_proba(X_test_tfidf)[:, 1]
    p_test_nn = phishnet_model.predict_proba(X_test_tab_scaled)
    p_test_tree = tree_model.predict_proba(X_test_tab_scaled)[:, 1]
    p_test_hyb = hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    p_test_moe = 0.50 * p_test_hyb + 0.30 * p_test_nn + 0.20 * p_test_nlp
    p_test_serving = SERVING_ALPHA * p_test_nlp + SERVING_BETA * p_test_hyb

    m_test_nlp = evaluate_predictions(y_test, p_test_nlp, 0.50)
    m_test_nn = evaluate_predictions(y_test, p_test_nn, 0.50)
    m_test_tree = evaluate_predictions(y_test, p_test_tree, 0.50)
    m_test_hyb = evaluate_predictions(y_test, p_test_hyb, 0.50)
    m_test_moe = evaluate_predictions(y_test, p_test_moe, best_thresh)
    m_test_serving = evaluate_predictions(y_test, p_test_serving, serving_thresh)

    print(f"{'Model Architecture':<34} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'FPR':<6} | {'ROC-AUC'}")
    print("-" * 100)
    print(f"{'1. Subword NLP Baseline':<34} | {m_test_nlp['accuracy']:<8.4f} | {m_test_nlp['precision']:<9.4f} | {m_test_nlp['recall_tpr']:<8.4f} | {m_test_nlp['f1_score']:<8.4f} | {m_test_nlp['fpr']:<6.4f} | {m_test_nlp['roc_auc']:.4f}")
    print(f"{'2. PyTorch PhishNet Deep NN (GPU)':<34} | {m_test_nn['accuracy']:<8.4f} | {m_test_nn['precision']:<9.4f} | {m_test_nn['recall_tpr']:<8.4f} | {m_test_nn['f1_score']:<8.4f} | {m_test_nn['fpr']:<6.4f} | {m_test_nn['roc_auc']:.4f}")
    print(f"{'3. Tabular LightGBM (30 Feat)':<34} | {m_test_tree['accuracy']:<8.4f} | {m_test_tree['precision']:<9.4f} | {m_test_tree['recall_tpr']:<8.4f} | {m_test_tree['f1_score']:<8.4f} | {m_test_tree['fpr']:<6.4f} | {m_test_tree['roc_auc']:.4f}")
    print(f"{'4. Hybrid Feature (LightGBM)':<34} | {m_test_hyb['accuracy']:<8.4f} | {m_test_hyb['precision']:<9.4f} | {m_test_hyb['recall_tpr']:<8.4f} | {m_test_hyb['f1_score']:<8.4f} | {m_test_hyb['fpr']:<6.4f} | {m_test_hyb['roc_auc']:.4f}")
    print(f"{'5. PhishGuard 2.0 MoE (Champion)':<34} | {m_test_moe['accuracy']:<8.4f} | {m_test_moe['precision']:<9.4f} | {m_test_moe['recall_tpr']:<8.4f} | {m_test_moe['f1_score']:<8.4f} | {m_test_moe['fpr']:<6.4f} | {m_test_moe['roc_auc']:.4f}")
    print(f"{'6. Serving Fusion (ml_service)':<34} | {m_test_serving['accuracy']:<8.4f} | {m_test_serving['precision']:<9.4f} | {m_test_serving['recall_tpr']:<8.4f} | {m_test_serving['f1_score']:<8.4f} | {m_test_serving['fpr']:<6.4f} | {m_test_serving['roc_auc']:.4f}")
    print("=" * 100)
    print("  Row 6 is the number to quote for the deployed API: it is the only row whose")
    print("  score formula matches what ml_service/models.py actually computes.")
    print("  Every row is measured on domains held out entirely from training.")

    # 5b. Out-of-corpus smoke benchmark.
    # The held-out split still comes from the same two CSVs and shares their quirks.
    # These 36 hand-labelled URLs come from neither corpus and are the check that
    # caught the previous release: it scored 80.6% here (30% FPR on benign) while
    # reporting 97.99% on its own test split, because the raw-text model had learned
    # the scheme artifact and returned p ~ 1.0 for essentially every real URL.
    # Treat a benign mean probability above ~0.5 as a failed run, not a passing one.
    SMOKE_BENIGN = [
        'google.com', 'wikipedia.org', 'github.com', 'sbi.co.in',
        'https://github.com/torvalds/linux', 'https://www.bbc.co.uk/news/world-us-canada-12345',
        'https://stackoverflow.com/questions/1234/how-to-parse-json',
        'https://en.wikipedia.org/wiki/Machine_learning', 'https://www.amazon.com/dp/B08N5WRWNW',
        'https://accounts.google.com/signin/v2/identifier',
        'https://login.microsoftonline.com/common/oauth2/authorize',
        'https://www.paypal.com/us/signin', 'https://mail.google.com/mail/u/0/',
        'https://www.reddit.com/r/programming/comments/abc123/',
        'https://news.ycombinator.com/item?id=38912345',
        'https://docs.python.org/3/library/os.path.html', 'https://www.irs.gov/payments',
        'https://www.linkedin.com/in/some-person-1234', 'https://drive.google.com/file/d/1a2b3c/view',
    ]
    SMOKE_MALICIOUS = [
        'https://amazon.com.security-update.ru/signin', 'http://192.168.1.1/paypal/login/verify.php',
        'http://paypa1-secure-login.tk/account/verify', 'https://appleid.apple.com.verify-account.xyz/login',
        'http://xn--80ak6aa92e.com/signin', 'http://secure-chase-bank.verify-now.top/update/billing',
        'http://netflix-billing-update.gq/account/payment',
        'https://microsoft-office365-login.duckdns.org/auth', 'http://wellsfargo.com-secure-id.cf/login.htm',
        'http://192.0.2.44:8080/wp-includes/paypal/websc.php',
        'https://binance-giveaway-claim.click/wallet/connect',
        'http://dhl-parcel-tracking-redelivery.buzz/pay', 'http://gooogle-account-recovery.icu/signin',
        'https://update-your-account-info-now.monster/verify/id', 'http://facebook.security-check.ml/login.php',
    ]
    smoke_urls = [normalize_url(u) for u in SMOKE_BENIGN + SMOKE_MALICIOUS]
    y_smoke = np.array([0] * len(SMOKE_BENIGN) + [1] * len(SMOKE_MALICIOUS))

    X_smoke_tab = scaler.transform(extract_features_batch(smoke_urls, chunk_size=64, n_jobs=1))
    X_smoke_tfidf = tfidf.transform(smoke_urls)
    X_smoke_hybrid = sp.hstack([X_smoke_tfidf, sp.csr_matrix(X_smoke_tab)], format='csr')
    p_smoke = (SERVING_ALPHA * nlp_model.predict_proba(X_smoke_tfidf)[:, 1]
               + SERVING_BETA * hybrid_model.predict_proba(X_smoke_hybrid)[:, 1])
    m_smoke = evaluate_predictions(y_smoke, p_smoke, serving_thresh)

    print("\n" + "-" * 100)
    print(f" OUT-OF-CORPUS SMOKE BENCHMARK ({len(smoke_urls)} hand-labelled URLs, serving fusion @ {serving_thresh})")
    print("-" * 100)
    print(f"  accuracy {m_smoke['accuracy']:.4f} | FPR {m_smoke['fpr']:.4f} | recall {m_smoke['recall_tpr']:.4f} "
          f"| ROC-AUC {m_smoke['roc_auc']:.4f}")
    print(f"  mean p on benign   : {p_smoke[y_smoke == 0].mean():.4f}  (previous release: 0.9918 - no separation)")
    print(f"  mean p on malicious: {p_smoke[y_smoke == 1].mean():.4f}")
    for u, yy, pp in zip(SMOKE_BENIGN + SMOKE_MALICIOUS, y_smoke, p_smoke):
        flag = 'ok  ' if (pp >= serving_thresh) == bool(yy) else 'MISS'
        print(f"    {flag} {'MAL' if yy else 'BEN'} p={pp:.3f}  {u[:72]}")
    print("-" * 100)

    # 6. Artifact Serialization
    print("\n" + "=" * 100)
    print(" [6/6] SERIALIZING MODEL ARTIFACTS & CREATING ZIP PACKAGE")
    print("=" * 100)

    joblib.dump(tree_model, os.path.join(SAVED_MODELS_DIR, 'tree_ensemble.joblib'))
    joblib.dump(hybrid_model, os.path.join(SAVED_MODELS_DIR, 'hybrid_lgbm.joblib'))
    joblib.dump(nlp_model, os.path.join(SAVED_MODELS_DIR, 'nlp_baseline.joblib'))
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'scaler.joblib'))
    joblib.dump(tfidf, os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.joblib'))

    # PyTorch Model Checkpoint and Wrapper.
    # phishnet_nn.pt is the durable artifact: plain tensors, loadable via
    # phishnet.load_phishnet() with no pickle involved. The joblib copy is kept only
    # for callers that still expect it, and it now pickles as
    # phishnet.PyTorchPhishNetClassifier rather than __main__.<class>, so it can
    # actually be read back outside this script.
    phishnet_model.save_checkpoint(os.path.join(SAVED_MODELS_DIR, 'phishnet_nn.pt'))
    joblib.dump(phishnet_model, os.path.join(SAVED_MODELS_DIR, 'phishnet_classifier.joblib'))

    # phishnet.py must travel with the artifacts or the joblib copy cannot be imported.
    import shutil as _shutil
    _shutil.copyfile(os.path.join(BASE_DIR, 'phishnet.py'),
                     os.path.join(SAVED_MODELS_DIR, 'phishnet.py'))

    # transformer_model.joblib is a legacy filename that ml_service falls back to when
    # nlp_baseline.joblib is missing; it must therefore hold the SAME kind of object as
    # nlp_baseline (a TF-IDF classifier), not the PhishNet wrapper. Writing the PhishNet
    # object here is what made the fallback path a latent crash.
    joblib.dump(nlp_model, os.path.join(SAVED_MODELS_DIR, 'transformer_model.joblib'))

    feature_importances = {}
    if hasattr(tree_model, 'feature_importances_'):
        imps = tree_model.feature_importances_
        tot = sum(imps) if sum(imps) > 0 else 1
        for n, v in zip(FEATURE_NAMES, imps):
            feature_importances[n] = round(float((v / tot) * 100), 2)

    report = {
        'test_set_size': len(y_test),
        'split_strategy': 'GroupShuffleSplit on registered domain (no host shared across splits)',
        'url_normalization': NORMALIZATION_VERSION,
        'optimal_threshold': best_thresh,
        'serving_threshold': serving_thresh,
        'hardware_accelerator': f"{DEVICE} ({GPU_NAME})",
        'baseline_nlp': m_test_nlp,
        'phishnet_deep_nn': m_test_nn,
        'tabular_lgbm': m_test_tree,
        'hybrid_lgbm': m_test_hyb,
        'champion_moe': m_test_moe,
        'serving_fusion': m_test_serving,
        'out_of_corpus_smoke': m_smoke,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'test_evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    threshold_cfg = {
        # Threshold for the 3-expert offline champion: 0.50*hybrid + 0.30*phishnet + 0.20*nlp
        'optimal_threshold': best_thresh,
        'weights': {'alpha_hybrid': 0.50, 'beta_phishnet': 0.30, 'gamma_nlp': 0.20},
        # Threshold for the 2-expert blend ml_service actually computes,
        # alpha*p_nlp + beta*p_hybrid, tuned at the default gating weights.
        # models.py should read THIS one.
        'serving_threshold': serving_thresh,
        'serving_weights': {'alpha_nlp': SERVING_ALPHA, 'beta_hybrid': SERVING_BETA},
        'url_normalization': NORMALIZATION_VERSION,
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
        # Named for the artifact it corresponds to. This is a linear TF-IDF classifier,
        # not a transformer - the old 'transformer_model' key implied otherwise.
        'nlp_baseline_model': m_test_nlp,
        'hybrid_moe_ensemble': m_test_moe,
        'serving_fusion': m_test_serving,
        'out_of_corpus_smoke': m_smoke,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_metrics_dict, f, indent=2)

    # Manifest: lets the serving side verify it is loading artifacts it can actually
    # interpret, instead of silently mixing a normalised model with un-normalised input
    # or an sklearn pickle from a different minor version.
    manifest = {
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'url_normalization': NORMALIZATION_VERSION,
        'split_strategy': 'group-by-registered-domain',
        'bare_domain_augment_fraction': BARE_DOMAIN_AUGMENT_FRACTION,
        'feature_names': list(FEATURE_NAMES),
        'n_features_tabular': len(FEATURE_NAMES),
        'n_features_tfidf': int(len(tfidf.vocabulary_)),
        'n_features_hybrid': int(X_train_hybrid.shape[1]),
        'library_versions': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__,
            'lightgbm': lgb.__version__ if HAS_LIGHTGBM else None,
            'torch': torch.__version__,
            'joblib': joblib.__version__
        },
        'artifacts': {
            'tree_ensemble.joblib': 'LGBMClassifier over the 30 tabular features',
            'hybrid_lgbm.joblib': 'LGBMClassifier over [tfidf | scaled tabular]',
            'nlp_baseline.joblib': 'SGDClassifier(log_loss) over tfidf',
            'transformer_model.joblib': 'legacy alias of nlp_baseline.joblib',
            'phishnet_classifier.joblib': 'phishnet.PyTorchPhishNetClassifier (needs phishnet.py)',
            'phishnet_nn.pt': 'weights-only PhishNet checkpoint (preferred; use phishnet.load_phishnet)',
            'scaler.joblib': 'StandardScaler for the tabular block',
            'tfidf_vectorizer.joblib': 'char_wb 3-5 gram TfidfVectorizer',
            'phishnet.py': 'class definitions required to unpickle phishnet_classifier.joblib'
        }
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("  • Saved all .joblib weights, PyTorch .pt weights, scalers, vectorizers, manifest and JSON metrics.")

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
    import argparse

    ap = argparse.ArgumentParser(
        description="PhishGuard 2.0 training pipeline (master dataset aware).")
    ap.add_argument('--quick-test', action='store_true',
                    help="End-to-end smoke run on 5,000 samples. Verifies ingestion, feature "
                         "extraction and model fitting; the resulting metrics are meaningless.")
    ap.add_argument('--max-samples', type=int, default=0,
                    help="Cap the training rows (0 = full dataset).")
    args = ap.parse_args()

    n = 5000 if args.quick_test else args.max_samples
    if args.quick_test:
        print("\n*** QUICK TEST MODE: 5,000 samples. Metrics from this run are NOT valid. ***\n")
    run_full_training(max_samples=n)
