# Reactor k-effective Prediction — Model Integration Guide

**Audience:** Backend engineering team integrating this model as a service.

---

## 1. Overview

This model predicts a nuclear reactor's **k-effective (k_eff)** — the neutron multiplication factor — from three physical inputs, and classifies the resulting reactor state.

**Inputs:**

| Field | Type | Meaning | Valid range |
|---|---|---|---|
| `enrichment_percent` | float | % of fissile U-235 in the fuel | 2.0 – 5.0 |
| `fuel_density` | float | how densely packed the fuel is | ~9.8 – 10.6 |
| `moderator_density` | float | how much neutron-slowing material is present | ~0.95 – 1.05 |

**Outputs:**

| Field | Type | Meaning |
|---|---|---|
| `predicted_k_eff` | float | the predicted neutron multiplication factor |
| `predicted_status` | string | `"Subcritical"` \| `"Critical"` \| `"Supercritical"` |

Status boundaries: `k_eff < 0.95` → Subcritical, `0.95 ≤ k_eff ≤ 1.05` → Critical, `k_eff > 1.05` → Supercritical.

Source notebook: `ML _ AI (Nuclear Team).ipynb`. (This folder also contains `AI_ML_Last_project.ipynb`, a separate, unrelated LSTM power-forecasting exercise — not part of this model, ignore it for this integration.)

## 2. Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sanity-check the model loads and predicts correctly
python predict.py 3.8
# -> {'enrichment_percent': 3.8, 'fuel_density': 10.3, 'moderator_density': 1.0,
#     'predicted_k_eff': 1.0054, 'predicted_status': 'Critical', 'model': 'Random Forest'}

python predict.py 4.2 10.4 1.0
# -> same shape, using your own fuel_density / moderator_density instead of the demo defaults

# 3. Build your backend service around predict.py (see "Integrating as a service" below)

# 4. To retrain the model from Reactor_Feasibility_Plot.csv:
python train_model.py
```

If step 2 works, the `.pkl` loads correctly and `reactor_k_eff_model.pkl` is ready to be wrapped in a service — everything past this point is about how to do that.

## 3. Models used

The regression target (`k_eff`) and the classification target (`status`) are handled by two separate models, both shipped in one artifact:

| Model | Role | Performance |
|---|---|---|
| **Random Forest Regressor** (100 trees) | Predicts continuous `k_eff` | R² = 0.998, RMSE = 0.0023 |
| **Decision Tree Classifier** (max_depth=4) | Predicts `Subcritical`/`Critical`/`Supercritical` | ~99% accuracy |

Random Forest was selected as the regressor after comparing against Linear Regression, a single Decision Tree, and degree-3 Polynomial Regression — it had the best R²/RMSE of the four because the enrichment→k_eff relationship is curved rather than linear, and ensembling smooths out noise. `enrichment_percent` is by far the dominant feature (~85–90% importance), with `fuel_density` and `moderator_density` contributing the remainder.

Both models are tree-based, so **no feature scaling is required** at inference time — raw input values go in as-is.

## 4. The model artifact

`reactor_k_eff_model.pkl` is a single joblib-serialized dict bundling both models plus the metadata a service needs to use them correctly:

```python
{
  "regressor": <fitted RandomForestRegressor>,
  "regressor_name": "Random Forest",
  "classifier": <fitted DecisionTreeClassifier>,
  "feature_names": ["enrichment_percent", "fuel_density", "moderator_density"],
  "status_thresholds": {"subcritical_max": 0.95, "supercritical_min": 1.05},
  "metrics": {"regressor_r2": 0.9977, "regressor_rmse": 0.0023, "classifier_accuracy": 0.99},
  "training_data_range": {
    "enrichment_percent": [2.0, 5.0],
    "fuel_density": [9.80, 10.60],
    "moderator_density": [0.95, 1.05]
  },
  "sklearn_version": "1.9.1",
  "trained_at_utc": "2026-09-17T00:55:35Z"
}
```

`model_metadata.json` is the same content minus the two model objects — read that instead of unpickling when you just need to inspect version/metrics/ranges.

## 5. Loading and calling the model

`predict.py` in this folder is the reference implementation — load the bundle once per process, not per request. It already exposes exactly what a service needs:

```python
# predict.py
from predict import load_bundle, predict_reactor_behavior

load_bundle()  # loads reactor_k_eff_model.pkl once; safe to call again, it caches

result = predict_reactor_behavior(
    enrichment_percent=3.8,
    fuel_density=10.3,
    moderator_density=1.0,
)
# -> {'enrichment_percent': 3.8, 'fuel_density': 10.3, 'moderator_density': 1.0,
#     'predicted_k_eff': 1.0054, 'predicted_status': 'Critical', 'model': 'Random Forest'}
```

Run it standalone from the command line to sanity-check the model before wiring it into anything:

```bash
python predict.py 3.8                  # uses demo defaults for fuel_density/moderator_density
python predict.py 4.2 10.4 1.0         # explicit enrichment, fuel_density, moderator_density
```

Internally it passes a `DataFrame` with `feature_names` (rather than a raw numpy array) to avoid sklearn's "X does not have valid feature names" warning, and raises `ValueError` if `enrichment_percent` is outside the training range — keep that behavior when you port the logic into the service.

## 6. Integrating as a service

Build the backend around `predict.py` rather than re-implementing the loading/prediction logic — import its two functions directly. Minimal FastAPI example:

```python
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import load_bundle, predict_reactor_behavior

app = FastAPI()

...
```

Run it:

```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```

Then call it:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"enrichment_percent": 3.8, "fuel_density": 10.3, "moderator_density": 1.0}'
# -> {"enrichment_percent":3.8,"fuel_density":10.3,"moderator_density":1.0,
#     "predicted_k_eff":1.0054,"predicted_status":"Critical","model":"Random Forest"}
```

Other things to carry into the real service, beyond this skeleton:

1. **Keep the range validation** (`predict.py` already raises `ValueError` — the example above turns that into an HTTP 400) instead of silently extrapolating; tree models don't degrade gracefully outside their training range.
2. **Pin `scikit-learn`/`joblib` versions** in the service's requirements to match `bundle["sklearn_version"]` (see `requirements.txt` in this folder) — unpickling sklearn estimators across major version gaps can break or emit warnings.
3. **Version the artifact** (e.g. filename suffix or a `model_version` field) so the API can report which model version served each response, once retraining starts happening.
4. **Add a couple of known-input regression tests** (e.g. enrichment ≈ 3.8% → `"Critical"`) so a future retrain that silently degrades the model gets caught in CI.
5. Loading a `.pkl` executes arbitrary Python on load — only load this file from the repo's own trusted build/release process, never from user-supplied input.

## 7. Retraining

`train_model.py` reproduces the full pipeline end to end: load `Reactor_Feasibility_Plot.csv` → train all four regressor candidates → pick the best by R² → train the classifier → save `reactor_k_eff_model.pkl` + `model_metadata.json`.

```bash
python train_model.py
```

Re-run this whenever `Reactor_Feasibility_Plot.csv` changes, then redeploy the regenerated `reactor_k_eff_model.pkl` to the service (bump the version per §6.3).

Note: the classifier's `Supercritical` class is small in the training data, so the split falls back to a non-stratified train/test split when needed (handled automatically in the script) — this is expected, not a bug.

## 8. Files in this folder

| File | Purpose |
|---|---|
| `Reactor_Feasibility_Plot.csv` | Training data (enrichment, densities, k_eff, status) |
| `ML _ AI (Nuclear Team).ipynb` | Original exploratory notebook — source of truth for modeling decisions |
| `AI_ML_Last_project.ipynb` | Unrelated SMR time-series exercise — not part of this model |
| `NUCLEAR TEAM PRESENTATION.docx` | Original team presentation explaining goals/findings |
| `train_model.py` | Reproducible script: data → trained models → `reactor_k_eff_model.pkl` |
| `predict.py` | Reference inference wrapper — import `load_bundle`/`predict_reactor_behavior` directly into the service |
| `reactor_k_eff_model.pkl` | Generated model bundle (regressor + classifier + metadata) |
| `model_metadata.json` | Human-readable mirror of the bundle's metadata (no model binaries) |
| `requirements.txt` | Pinned versions used to train/serve the model (`pandas`, `numpy`, `scikit-learn`, `joblib`) |
| `readme.md` | This document |
| `NOTES.md` | Internal observations, data caveats, improvement ideas, and hosting considerations — read this before deciding how much to invest in retraining/infra |
