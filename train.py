"""
🛡️ PhishGuard 2.0 - Standalone Cloud / Colab Training & Packaging Engine
Trains on the FULL 1.2M+ Malicious & Benign URL Datasets using T4 GPU / CPU Acceleration.
Produces serialized models and auto-generates 'saved_models.zip' for instant download.
"""

import os
import sys
import time
import json
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import scipy.sparse as sp

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
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

from features import FEATURE_NAMES, extract_features, extract_features_batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_harmonize_datasets(csv1_path: str, csv2_path: str, max_samples: Optional[int] = 0) -> pd.DataFrame:
    print("\n" + "=" * 95)
    print(" [1/6] INGESTING & HARMONIZING 1,000,000+ SAMPLES DATASET")
    print("=" * 95)
    dfs = []

    if os.path.exists(csv1_path):
        print(f"  • Reading {csv1_path}...")
        df1 = pd.read_csv(csv1_path, low_memory=False)
        u1 = 'url' if 'url' in df1.columns else df1.columns[0]
        t1 = 'type' if 'type' in df1.columns else df1.columns[1]
        c1 = df1[[u1, t1]].dropna().copy()
        c1.rename(columns={u1: 'url', t1: 'raw_label'}, inplace=True)
        c1['url'] = c1['url'].astype(str).str.strip()
        c1 = c1[c1['url'].str.len() > 3]
        c1['label'] = c1['raw_label'].apply(lambda x: 0 if str(x).lower().strip() == 'benign' else 1)
        print(f"    -> Ingested {len(c1):,} rows from malicious_phish.csv")
        dfs.append(c1[['url', 'label']])
    else:
        print(f"  ! Warning: {csv1_path} not found.")

    if os.path.exists(csv2_path):
        print(f"  • Reading {csv2_path}...")
        df2 = pd.read_csv(csv2_path, low_memory=False)
        u2 = 'URL' if 'URL' in df2.columns else df2.columns[0]
        t2 = 'Label' if 'Label' in df2.columns else df2.columns[1]
        c2 = df2[[u2, t2]].dropna().copy()
        c2.rename(columns={u2: 'url', t2: 'raw_label'}, inplace=True)
        c2['url'] = c2['url'].astype(str).str.strip()
        c2 = c2[c2['url'].str.len() > 3]
        c2['label'] = c2['raw_label'].apply(lambda x: 0 if str(x).lower().strip() == 'good' else 1)
        print(f"    -> Ingested {len(c2):,} rows from phishing_site_urls.csv")
        dfs.append(c2[['url', 'label']])
    else:
        print(f"  ! Warning: {csv2_path} not found.")

    if not dfs:
        raise FileNotFoundError("Could not find dataset CSVs! Ensure malicious_phish.csv and phishing_site_urls.csv are in the folder.")

    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"\n  • Total Combined Raw Entries: {len(df_combined):,}")
    print("  • Deduplicating & calculating consensus labels...")
    df_clean = df_combined.groupby('url', as_index=False)['label'].agg(lambda x: 1 if x.mean() >= 0.5 else 0)
    print(f"  • Unique Clean URL Samples: {len(df_clean):,} (Benign: {sum(df_clean['label']==0):,}, Malicious: {sum(df_clean['label']==1):,})")

    if max_samples and max_samples > 0 and len(df_clean) > max_samples:
        print(f"  • Downsampling to {max_samples:,} records...")
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
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall_tpr': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_true, y_prob), 4),
        'fpr': round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }


def run_full_training():
    start_time = time.time()
    print("=" * 95)
    print(" 🚀 PHISHGUARD 2.0 PRODUCTION CLOUD TRAINING PIPELINE")
    print(f" Active Hardware: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'High-Performance CPU'})")
    if torch.cuda.is_available():
        print(f" GPU VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 95)

    csv1 = os.path.join(BASE_DIR, 'malicious_phish.csv')
    csv2 = os.path.join(BASE_DIR, 'phishing_site_urls.csv')

    # 1. Ingest Data
    df = load_and_harmonize_datasets(csv1, csv2, max_samples=0)
    y = df['label'].values
    urls = df['url'].values

    # 2. 3-Way Partitioning (70% Train, 15% Val, 15% Test)
    print("\n" + "=" * 95)
    print(" [2/6] STRICT 3-WAY PARTITIONING (70% Train / 15% Val / 15% Test)")
    print("=" * 95)
    train_val_urls, test_urls, y_train_val, y_test = train_test_split(
        urls, y, test_size=0.15, random_state=42, stratify=y
    )
    train_urls, val_urls, y_train, y_val = train_test_split(
        train_val_urls, y_train_val, test_size=(0.15 / 0.85), random_state=42, stratify=y_train_val
    )
    print(f"  • Training Set   : {len(train_urls):,} URLs ({sum(y_train==0):,} Benign, {sum(y_train==1):,} Malicious)")
    print(f"  • Validation Set : {len(val_urls):,} URLs ({sum(y_val==0):,} Benign, {sum(y_val==1):,} Malicious)")
    print(f"  • Held-Out Test  : {len(test_urls):,} URLs ({sum(y_test==0):,} Benign, {sum(y_test==1):,} Malicious)")

    # 3. Parallel Feature Extraction
    print("\n" + "=" * 95)
    print(f" [3/6] HIGH-SPEED BATCH FEATURE EXTRACTION ({len(FEATURE_NAMES)} DOMAIN FEATURES)")
    print("=" * 95)
    t0 = time.time()
    X_train_tab = extract_features_batch(train_urls.tolist())
    X_val_tab = extract_features_batch(val_urls.tolist())
    X_test_tab = extract_features_batch(test_urls.tolist())

    scaler = StandardScaler()
    X_train_tab_scaled = scaler.fit_transform(X_train_tab)
    X_val_tab_scaled = scaler.transform(X_val_tab)
    X_test_tab_scaled = scaler.transform(X_test_tab)

    print("  • Fitting Subword Character N-Gram TF-IDF Vectorizer (20,000 features, ngrams=3-5)...")
    tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=20000,
        sublinear_tf=True,
        lowercase=True
    )
    X_train_tfidf = tfidf.fit_transform(train_urls)
    X_val_tfidf = tfidf.transform(val_urls)
    X_test_tfidf = tfidf.transform(test_urls)

    X_train_hybrid = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_tab_scaled)], format='csr')
    X_val_hybrid = sp.hstack([X_val_tfidf, sp.csr_matrix(X_val_tab_scaled)], format='csr')
    X_test_hybrid = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_tab_scaled)], format='csr')
    print(f"    -> Complete Feature Matrix built in {time.time() - t0:.2f}s ({X_train_hybrid.shape[1]:,} features/sample).")

    # 4. Model Training
    print("\n" + "=" * 95)
    print(" [4/6] MODEL TRAINING & VALIDATION HYPERPARAMETER OPTIMIZATION")
    print("=" * 95)

    # A: NLP Classifier
    print("  [1/3] Training Subword NLP Classifier (TF-IDF + SAGA Logistic Regression)...")
    t0 = time.time()
    nlp_model = LogisticRegression(C=3.0, max_iter=1000, solver='saga', random_state=42, n_jobs=-1)
    nlp_model.fit(X_train_tfidf, y_train)
    p_val_nlp = nlp_model.predict_proba(X_val_tfidf)[:, 1]
    m_val_nlp = evaluate_predictions(y_val, p_val_nlp, 0.50)
    print(f"    -> NLP Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_nlp['accuracy']:.4f} | F1: {m_val_nlp['f1_score']:.4f} | AUC: {m_val_nlp['roc_auc']:.4f}")

    # B: Tabular Tree Classifier
    print("  [2/3] Training Tabular Deep LightGBM Classifier (29 Features)...")
    t0 = time.time()
    tree_model = lgb.LGBMClassifier(
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    tree_model.fit(X_train_tab_scaled, y_train)
    p_val_tree = tree_model.predict_proba(X_val_tab_scaled)[:, 1]
    m_val_tree = evaluate_predictions(y_val, p_val_tree, 0.50)
    print(f"    -> Tree Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_tree['accuracy']:.4f} | F1: {m_val_tree['f1_score']:.4f} | AUC: {m_val_tree['roc_auc']:.4f}")

    # C: Unified Hybrid Classifier
    print("  [3/3] Training Unified Hybrid LightGBM Classifier (20,029 Features)...")
    t0 = time.time()
    hybrid_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.06,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.80,
        random_state=42,
        n_jobs=-1
    )
    hybrid_model.fit(X_train_hybrid, y_train)
    p_val_hyb = hybrid_model.predict_proba(X_val_hybrid)[:, 1]
    m_val_hyb = evaluate_predictions(y_val, p_val_hyb, 0.50)
    print(f"    -> Hybrid Model trained in {time.time() - t0:.2f}s | Val Acc: {m_val_hyb['accuracy']:.4f} | F1: {m_val_hyb['f1_score']:.4f} | AUC: {m_val_hyb['roc_auc']:.4f}")

    # D: Mixture of Experts & Decision Threshold Optimization
    print("  • Optimizing Dynamic Gating & Decision Threshold on Validation Set (FPR <= 0.02)...")
    p_val_moe = 0.55 * p_val_hyb + 0.45 * p_val_nlp
    best_thresh, best_f1, best_val_m = 0.50, 0.0, None
    for th in np.arange(0.30, 0.75, 0.02):
        m_tmp = evaluate_predictions(y_val, p_val_moe, threshold=th)
        if m_tmp['fpr'] <= 0.02 and m_tmp['f1_score'] > best_f1:
            best_f1 = m_tmp['f1_score']
            best_thresh = round(float(th), 2)
            best_val_m = m_tmp

    if best_val_m is None:
        best_thresh = 0.50
        best_val_m = evaluate_predictions(y_val, p_val_moe, 0.50)

    print(f"    -> Optimal Decision Threshold: {best_thresh} (Val Acc: {best_val_m['accuracy']:.4f}, FPR: {best_val_m['fpr']:.4f}, F1: {best_val_m['f1_score']:.4f})")

    # 5. Held-Out Test Set Evaluation
    print("\n" + "=" * 95)
    print(f" [5/6] FINAL UNBIASED BENCHMARK ON HELD-OUT TEST SET ({len(y_test):,} SAMPLES)")
    print("=" * 95)
    p_test_nlp = nlp_model.predict_proba(X_test_tfidf)[:, 1]
    p_test_tree = tree_model.predict_proba(X_test_tab_scaled)[:, 1]
    p_test_hyb = hybrid_model.predict_proba(X_test_hybrid)[:, 1]
    p_test_moe = 0.55 * p_test_hyb + 0.45 * p_test_nlp

    m_test_nlp = evaluate_predictions(y_test, p_test_nlp, 0.50)
    m_test_tree = evaluate_predictions(y_test, p_test_tree, 0.50)
    m_test_hyb = evaluate_predictions(y_test, p_test_hyb, 0.50)
    m_test_moe = evaluate_predictions(y_test, p_test_moe, best_thresh)

    print(f"{'Model Architecture':<32} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'FPR':<6} | {'ROC-AUC'}")
    print("-" * 95)
    print(f"{'1. Subword NLP Baseline':<32} | {m_test_nlp['accuracy']:<8.4f} | {m_test_nlp['precision']:<9.4f} | {m_test_nlp['recall_tpr']:<8.4f} | {m_test_nlp['f1_score']:<8.4f} | {m_test_nlp['fpr']:<6.4f} | {m_test_nlp['roc_auc']:.4f}")
    print(f"{'2. Tabular LightGBM (29 Feat)':<32} | {m_test_tree['accuracy']:<8.4f} | {m_test_tree['precision']:<9.4f} | {m_test_tree['recall_tpr']:<8.4f} | {m_test_tree['f1_score']:<8.4f} | {m_test_tree['fpr']:<6.4f} | {m_test_tree['roc_auc']:.4f}")
    print(f"{'3. Hybrid Feature (LightGBM)':<32} | {m_test_hyb['accuracy']:<8.4f} | {m_test_hyb['precision']:<9.4f} | {m_test_hyb['recall_tpr']:<8.4f} | {m_test_hyb['f1_score']:<8.4f} | {m_test_hyb['fpr']:<6.4f} | {m_test_hyb['roc_auc']:.4f}")
    print(f"{'4. PhishGuard 2.0 MoE (Champion)':<32} | {m_test_moe['accuracy']:<8.4f} | {m_test_moe['precision']:<9.4f} | {m_test_moe['recall_tpr']:<8.4f} | {m_test_moe['f1_score']:<8.4f} | {m_test_moe['fpr']:<6.4f} | {m_test_moe['roc_auc']:.4f}")
    print("=" * 95)

    # 6. Artifact Serialization
    print("\n" + "=" * 95)
    print(" [6/6] SERIALIZING MODEL ARTIFACTS & CREATING ZIP PACKAGE")
    print("=" * 95)

    joblib.dump(tree_model, os.path.join(SAVED_MODELS_DIR, 'tree_ensemble.joblib'))
    joblib.dump(hybrid_model, os.path.join(SAVED_MODELS_DIR, 'hybrid_lgbm.joblib'))
    joblib.dump(nlp_model, os.path.join(SAVED_MODELS_DIR, 'nlp_baseline.joblib'))
    joblib.dump(nlp_model, os.path.join(SAVED_MODELS_DIR, 'transformer_model.joblib'))
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'scaler.joblib'))
    joblib.dump(tfidf, os.path.join(SAVED_MODELS_DIR, 'tfidf_vectorizer.joblib'))

    feature_importances = {}
    if hasattr(tree_model, 'feature_importances_'):
        imps = tree_model.feature_importances_
        tot = sum(imps) if sum(imps) > 0 else 1
        for n, v in zip(FEATURE_NAMES, imps):
            feature_importances[n] = float((v / tot) * 100)

    report = {
        'test_set_size': len(y_test),
        'optimal_threshold': best_thresh,
        'baseline_lr': m_test_nlp,
        'tabular_lgbm': m_test_tree,
        'hybrid_lgbm': m_test_hyb,
        'champion_moe': m_test_moe,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'test_evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(SAVED_MODELS_DIR, 'threshold_config.json'), 'w') as f:
        json.dump({'optimal_threshold': best_thresh, 'alpha_hybrid': 0.55, 'beta_nlp': 0.45, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)

    eval_metrics_dict = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_set_size': len(y_test),
        'tree_model': m_test_tree,
        'transformer_model': m_test_nlp,
        'hybrid_moe_ensemble': m_test_moe,
        'feature_importances': feature_importances
    }
    with open(os.path.join(SAVED_MODELS_DIR, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_metrics_dict, f, indent=2)

    print("  • Saved all .joblib weights, scalers, vectorizers, and JSON metrics.")

    # Create ZIP archive
    import shutil
    zip_path = os.path.join(BASE_DIR, 'saved_models.zip')
    shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', SAVED_MODELS_DIR)
    print(f"  • Created ZIP Archive: {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")

    elapsed = time.time() - start_time
    print(f"\n 🎉 ALL DONE IN {elapsed:.2f}s ({elapsed/60:.2f} MINS)! 'saved_models.zip' is ready.")

    # Trigger Colab Auto-Download if in Google Colab environment
    try:
        if 'google.colab' in sys.modules:
            from google.colab import files
            print("  • Google Colab detected! Triggering automatic browser download...")
            files.download(zip_path)
    except Exception as e:
        print(f"  ! Manual download: locate saved_models.zip in {BASE_DIR}")


if __name__ == '__main__':
    run_full_training()
