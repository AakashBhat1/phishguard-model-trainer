> **Links:** [[../Welcome]] | [[Style DNA]]

# PhishGuard 2.0 — ML Scoring Diagnosis

**Date:** 2026-08-21
**Status:** `REJECTED` / DEFERRED — findings recorded only, **no code changed**. Pick up tomorrow.
**Trigger:** `/explain` returned `risk_score: 100, verdict: PHISHING` for the benign URL `www.sg.com`,
while all 8 SHAP contributions were labelled `risk_reducing`.

---

## TL;DR

Three separate problems stacked on top of each other. The `100 / PHISHING` is the least serious one.

1. The pasted response came from a **stale uvicorn process** — current working copy returns `4 / SAFE`.
2. `4 / SAFE` is **also wrong** — it comes from a hardcoded suppression hack, not from the model.
   The model's actual belief for `www.sg.com` is `p = 1.0000` (certain phishing).
3. Root cause is **two dataset artifacts** (scheme leakage + bare-domain OOD), not a broken model.
4. Bonus: the SHAP layer is **decorative** — it never touches the model.

---

## Finding 1 — Stale server process

The false-positive suppression block in `ml_service/models.py:262-278` is **uncommitted**:

```
git show HEAD:ml_service/models.py | grep -c "has_zero_threat_signals"   ->  0
git diff       ml_service/models.py | grep -c "has_zero_threat_signals"   ->  3
```

The running uvicorn was started before that edit and `--reload` did not pick it up.
**Action:** restart the ML service. The `100` disappears on its own.

---

## Finding 2 — `4 / SAFE` is a band-aid, not a prediction

Raw model output for `www.sg.com`:

```
p_nlp    = 1.0000
p_hybrid = 0.9999
p_final  = 0.0400   <-- forced by suppression block, unrelated to model belief
```

The suppression gate only fires when a URL has *literally zero* threat signals
AND (`path_depth <= 1` OR `entropy < 3.8`). Ordinary benign URLs fail that gate:

| URL                                              | score | verdict   | correct? |
|--------------------------------------------------|-------|-----------|----------|
| `www.sg.com`                                     |   4   | SAFE      | by hack  |
| `google.com`                                     |   4   | SAFE      | by hack  |
| `https://github.com/torvalds/linux`              |   4   | SAFE      | by hack  |
| `https://www.bbc.co.uk/news/world-us-canada-12345` |  98 | PHISHING  | **WRONG**|
| `http://bit.ly/3xKd9`                            |   4   | SAFE      | **WRONG**|
| `https://amazon.com.security-update.ru/signin`   | 100   | PHISHING  | by luck  |

---

## Finding 3 — Root cause: two dataset artifacts

The models are **fine** on in-distribution data. Sampled 800 rows from `phishing_site_urls.csv`:

```
hybrid  AUC=0.9336  acc@0.5=0.9213  mean_p|benign=0.1512  mean_p|mal=0.9541
nlp     AUC=0.9319  acc@0.5=0.8950  mean_p|benign=0.1536  mean_p|mal=0.8388
tree    AUC=0.9300  acc@0.5=0.8525  mean_p|benign=0.2659  mean_p|mal=0.8626
```

Artifact dimensions are all self-consistent (scaler 30, tfidf 20000, hybrid 20030, tree 30),
so this is NOT a shape/order mismatch.

### 3a. Scheme leakage (the big one)

68.4% of malicious URLs carry `http://` / `https://`; only ~4% of benign do
(0.0% in `phishing_site_urls.csv`). The `char_wb` TF-IDF learned **"has a scheme => phishing"**.

```
mp-BENIGN      has_scheme=0.083  n=428103
mp-MALICIOUS   has_scheme=0.684  n=223088
psu-GOOD       has_scheme=0.000  n=392924
psu-BAD        has_scheme=0.001  n=156422
```

Adding a scheme alone flips the verdict:

| URL                                             | bare  | with `https://` |
|-------------------------------------------------|-------|-----------------|
| `wikipedia.org`                                 | 0.459 | **0.980**       |
| `bbc.co.uk/news/world`                          | 0.250 | **0.944**       |
| `github.com/torvalds/linux`                     | 0.573 | **0.970**       |
| `stackoverflow.com/questions/1234/how-to-parse` | 0.020 | 0.166           |

### 3b. Bare domains are out-of-distribution

Only **63 of 392,924** benign URLs (0.02%) have no path. Benign training data is almost
entirely deep paths (median length 40). A bare domain like `www.sg.com` is territory the
model has never seen labelled benign, so it defaults to 1.0.

### 3c. Reported metrics are inflated

`saved_models/evaluation_metrics.json` claims 94-96% accuracy / 0.99 ROC-AUC on a 106k
test set — but that split carries the same leakage, so the number does not transfer to
real-world input.

---

## Finding 4 — The SHAP values are decorative

`ml_service/xai_explainer.py:47-63` is a hardcoded lookup table of arithmetic on raw
features. It **never touches the LGBM model**, despite the docstring claiming "TreeSHAP".

Consequences:
- Verdict comes from the model; attributions come from unrelated hand-tuned constants.
  Nothing forces them to agree -> all 8 can read `risk_reducing` under a `PHISHING` verdict.
- `top_risks` ends up empty, so the SOC summary renders literally:
  `"Key driving factors include ."`
- `importance_percentage` is just `abs(shap * 100)`. The 8 values in the reported output
  sum to **48.8%**, not 100% — it is not a percentage of anything.

---

## Proposed fixes (NOT IMPLEMENTED — for tomorrow)

Ordered by payoff:

1. **Normalize URLs identically at train + inference** — strip scheme, lowercase,
   strip leading `www.`. ~5 lines in `features.py` + a retrain. Kills the leakage.
2. **Add bare-domain benign samples** (Tranco / Majestic top-1M) so short domains
   are no longer OOD.
3. **Replace the fake SHAP** with a real `shap.TreeExplainer` on `hybrid_lgbm`.
   The model already exists; this permanently fixes the verdict/attribution contradiction.
4. **Delete the suppression block** in `models.py` once 1-3 land — it currently hides
   the failure rather than fixing it.
5. **Re-evaluate** on a split that is not scheme-separable. Expect the headline number
   to drop; that drop is the honest one.

Also worth fixing while in there:
- `main.py` `/health` reports `"supported_features": 29`, actual `len(FEATURE_NAMES)` is **30**.
- sklearn version mismatch on load: artifacts pickled with **1.6.1**, environment has **1.5.2**
  (`InconsistentVersionWarning` on every model load). Pin the version.

---

## Reproduction commands

```bash
cd C:/dev/Phishguard/ml_service

# current verdict for the reported URL
python -W ignore -c "from models import detector; print(detector.predict('www.sg.com'))"

# raw model belief vs suppressed score
python -W ignore -c "
from models import detector
r = detector.predict('www.sg.com')
print('p_nlp', r['p_transformer'], 'p_hybrid', r['p_tree'], 'final', r['risk_score'])"

# scheme-leakage demo
python -W ignore -c "
import joblib
tf = joblib.load('saved_models/tfidf_vectorizer.joblib')
nl = joblib.load('saved_models/nlp_baseline.joblib')
print(nl.predict_proba(tf.transform(['wikipedia.org','https://wikipedia.org']))[:,1])"
```
