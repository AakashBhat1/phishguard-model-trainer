# 🛡️ PhishGuard 2.0 Model Trainer

> **High-Throughput NVIDIA Tesla T4 GPU & Cloud Training Engine**  
> Ingests, harmonizes, and trains production ensemble ML & Deep Learning models on **1,200,000+ malicious & benign URLs** with FP16 Tensor Core acceleration, 30-D heuristic feature engineering, and dynamic Mixture-of-Experts (MoE) gating.

---

## ⚡ Architecture & Models

```
                                  ┌──────────────────────────┐
                                  │   Raw URL Input Stream   │
                                  └─────────────┬────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 │                                                             │
                 ▼                                                             ▼
   ┌───────────────────────────┐                                 ┌───────────────────────────┐
   │  30-D Heuristic Signals   │                                 │ Subword Char N-Grams TFIDF│
   │       (features.py)       │                                 │      (20,000 Vocab)       │
   └─────────────┬─────────────┘                                 └─────────────┬─────────────┘
                 │                                                             │
        ┌────────┴────────┬─────────────────────────────┐                      │
        │                 │                             │                      │
        ▼                 ▼                             ▼                      ▼
┌──────────────┐  ┌───────────────┐            ┌─────────────────┐    ┌─────────────────┐
│Tabular Tree  │  │PyTorch Deep NN│            │ Hybrid LightGBM │    │ Subword NLP     │
│ (LightGBM)   │  │ (PhishNet-T4) │            │ (30-D + TF-IDF) │    │  (SGD Log-Loss) │
└───────┬──────┘  └───────┬───────┘            └────────┬────────┘    └────────┬────────┘
        │                 │                             │                      │
        └─────────────────┼─────────────────────────────┴──────────────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   Dynamic MoE Ensemble    │
            │ (FPR <= 1.5% Calibration) │
            └─────────────┬─────────────┘
                          │
                          ▼
            ┌───────────────────────────┐
            │   Phishing Risk Score     │
            │ (0.0 - 1.0) & Threat Intel│
            └───────────────────────────┘
```

1. **Cybersecurity Feature Extractor (`features.py`)**:
   - 30 domain-specific heuristic signals per URL (Shannon entropy, Levenshtein brand distance, IP-in-host, high-risk TLDs, punycode/IDN spoofing, vowel-to-consonant ratios, token diversity).
   - Optimized with pre-compiled regular expressions and length-pruned string matching.
2. **PyTorch Deep Residual PhishNet (`PhishNetDeep`)**:
   - Multi-layer deep residual architecture with BatchNorm, SiLU non-linearities, and Dropout.
   - Accelerated for NVIDIA Tesla T4 GPU with `torch.amp` FP16 Tensor Core mixed precision and pinned memory DataLoaders.
3. **Tabular & Hybrid LightGBM Boosted Trees**:
   - CUDA/GPU accelerated gradient boosted decision trees with automatic fallback to high-speed CPU multi-threading.
4. **Subword Character N-Gram Classifier**:
   - 20,000 subword character n-gram TF-IDF representations trained with high-speed calibrated log-loss SGD.
5. **Mixture-of-Experts (MoE) Decision Gating**:
   - Precision-calibrated ensemble combining Hybrid LightGBM, PhishNet Deep NN, and Subword NLP with optimal decision thresholding (FPR $\le$ 1.5%).

---

## 📁 Repository Structure

```
phishguard_training_pack/
├── malicious_phish.csv         # 651,000+ labeled URL dataset
├── phishing_site_urls.csv      # 549,000+ labeled URL dataset
├── features.py                 # 30-D heuristic feature extractor
├── train.py                    # Standalone full GPU/CPU training & export engine
├── train.ipynb                 # Interactive Google Colab T4 GPU notebook
├── README.md                   # Complete documentation
└── .gitignore
```

---

## 🚀 How to Run on Google Colab (NVIDIA T4 GPU)

### Method A: Interactive Google Colab Notebook (`train.ipynb`) — **Recommended**
1. Upload this repository folder (or clone it) to Google Colab.
2. Ensure GPU runtime is selected:
   - Click **Runtime** → **Change runtime type**
   - Under **Hardware accelerator**, select **T4 GPU**
   - Click **Save**
3. Open `train.ipynb` and click **Runtime** → **Run all**.
4. The notebook will:
   - Validate your T4 GPU and VRAM (15.36 GB GDDR6).
   - Train across the full 1.2M+ dataset in ~3-5 minutes.
   - Display evaluation metrics, comparison charts, and confusion matrix data.
   - Automatically trigger download of `saved_models.zip`.

### Method B: Terminal / Single Code Cell in Colab
In a Colab cell, execute:
```bash
# 1. Configure OpenCL / GPU acceleration drivers for Tesla T4 GPU
!apt-get install -y -qq ocl-icd-libopencl1 opencl-headers clinfo libboost-all-dev > /dev/null 2>&1
!mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd 2>/dev/null || true

# 2. Install dependencies
!pip install torch torchvision lightgbm scikit-learn pandas numpy scipy joblib tqdm matplotlib seaborn -q

# 3. Run full training with T4 GPU acceleration
!python train.py
```

### Method C: Local Execution
```bash
# Clone the repository
git clone https://github.com/AakashBhat1/phishguard-model-trainer.git
cd phishguard-model-trainer

# Install dependencies
pip install torch torchvision lightgbm scikit-learn pandas numpy scipy joblib tqdm

# Execute training
python train.py
```

---

## 📊 Output Artifacts in `saved_models.zip`

Once training finishes, all artifacts are saved to `saved_models/` and bundled into `saved_models.zip`:

| Artifact File | Description |
| :--- | :--- |
| `tree_ensemble.joblib` | Trained Tabular 30-D LightGBM classifier |
| `hybrid_lgbm.joblib` | Trained Unified Hybrid LightGBM model (20,030 features) |
| `nlp_baseline.joblib` | Trained Subword TF-IDF NLP model |
| `phishnet_nn.pt` | PyTorch Deep Residual Neural Network weights (`state_dict`) |
| `transformer_model.joblib` | Joblib-compatible PyTorch PhishNet inference wrapper |
| `scaler.joblib` | Fitted `StandardScaler` for 30-D tabular domain features |
| `tfidf_vectorizer.joblib` | Fitted 20,000 subword character n-gram vectorizer |
| `threshold_config.json` | Optimal MoE weights ($\alpha, \beta, \gamma$) & decision threshold |
| `test_evaluation_report.json` | Detailed benchmark metrics on held-out test set |
| `evaluation_metrics.json` | Extended model metrics and feature importance percentages |
| `saved_models.zip` | Complete auto-packaged bundle for instant download |

---

## 🔒 License
MIT License
