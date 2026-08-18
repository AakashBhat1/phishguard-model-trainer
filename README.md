# 🛡️ PhishGuard Model Trainer

> High-throughput GPU/CPU training pipeline for the PhishGuard cybersecurity platform. Ingests, harmonizes, and trains ensemble ML models (LightGBM, PyTorch Neural Network, TF-IDF + Logistic Regression) on **1,200,000+ malicious & benign URLs**.

---

## 📁 Repository Structure

```
├── malicious_phish.csv       # 651,000+ labeled URL dataset
├── phishing_site_urls.csv    # 549,000+ labeled URL dataset
├── features.py               # 29-dimension cybersecurity URL feature extractor
├── train.py                  # Standalone full training, evaluation & packaging script
├── train.ipynb               # 1-click Google Colab interactive notebook
└── .gitignore
```

---

## ⚡ Models & Architecture

1. **Cybersecurity Feature Extractor (`features.py`)**:
   - Computes 29 domain-specific heuristic signals per URL (entropy, IP-in-host, suspicious TLDs, punycode/IDN spoofing, lexical ratios, brand targeting tokens).
2. **LightGBM Boosted Classifier**:
   - Gradient boosted tree ensemble trained on scaled 29-D feature vectors for ultra-fast sub-millisecond inference.
3. **PyTorch Deep Neural Network (PhishNet)**:
   - Multi-layer dense residual architecture with BatchNorm and Dropout for robust generalization.
4. **TF-IDF + Logistic Regression**:
   - Character n-gram lexical vectorizer capturing raw URL structural anomalies.

---

## 🚀 How to Run

### Method A: Google Colab (Recommended for Free GPU)
1. Clone or upload this repository to Google Colab.
2. In Colab, execute:
   ```bash
   pip install torch lightgbm scikit-learn pandas numpy scipy joblib tqdm -q
   python train.py
   ```
3. The script trains all models across the full 1.2M dataset, prints comprehensive classification metrics (Accuracy, ROC-AUC, F1-Score), packages artifacts into `saved_models.zip`, and triggers an auto-download.

### Method B: Interactive Notebook (`train.ipynb`)
1. Upload `train.ipynb` along with the datasets to Google Colab.
2. Select **Runtime** → **Run all**.

### Method C: Local Execution
```bash
# Clone the repository
git clone https://github.com/AakashBhat1/phishguard-model-trainer.git
cd phishguard-model-trainer

# Install dependencies
pip install torch lightgbm scikit-learn pandas numpy scipy joblib tqdm

# Execute training
python train.py
```

---

## 📊 Training Output Artifacts
Once training finishes, the following artifacts are generated in `saved_models/`:
- `lightgbm_model.joblib` / `hgb_model.joblib`
- `phishnet_nn.pt`
- `tfidf_vectorizer.joblib`
- `lr_tfidf_model.joblib`
- `scaler.joblib`
- `metrics.json`
- `saved_models.zip` (Auto-packaged bundle)

---

## 🔒 License
MIT License
