# Banking-Fraud_Detection-Model
A banking fraud detection model is a predictive algorithm or system designed to identify and prevent fraudulent activities within the banking and financial industry. Fraud detection is crucial for banks and financial institutions to protect themselves and their customers from various forms of financial fraud, including credit card fraud, identity theft, account takeover, and more. These models use data analytics, machine learning, and other techniques to detect unusual or suspicious patterns and behaviors that may indicate fraudulent transactions.
Here are the key components and features of a banking fraud detection model:

Data Sources: The model relies on a variety of data sources, including transaction data, customer profiles, historical fraud cases, and external data sources. This data is used to train the model and continuously update it.

Machine Learning Algorithms: Most modern fraud detection models leverage machine learning algorithms to analyze and identify patterns in the data. Common algorithms used for fraud detection include logistic regression, decision trees, random forests, support vector machines, and neural networks.

Features: The model considers numerous features or variables in the data to assess the likelihood of fraud. These features can include transaction amount, location, time of day, transaction frequency, customer behavior history, and more.

Anomaly Detection: One of the primary methods used in fraud detection is anomaly detection. The model learns the typical behavior of legitimate transactions and flags transactions that deviate significantly from this norm as potential fraud.

Rules and Thresholds: In addition to machine learning, rules-based systems can be integrated into the model. These rules are based on industry knowledge and can include predefined thresholds for certain transaction characteristics, like unusually large transactions or transactions from high-risk countries.

Real-time Monitoring: Fraud detection models often operate in real-time, allowing them to assess transactions as they occur. Real-time monitoring enables immediate action to be taken if suspicious activity is detected.

Scalability: The model should be scalable to handle large volumes of transactions, especially for major financial institutions that process millions of transactions daily.

Integration with Fraud Alerts: When potential fraud is detected, the model can trigger fraud alerts, notifying bank personnel or customers to investigate further.

Continuous Learning: Like flight delay models, fraud detection models need continuous learning. New fraud patterns and techniques emerge, and the model must adapt to stay effective.

Customer Communication: Banks often have protocols for contacting customers when fraud is suspected, whether it's through automated messages or contact with customer service representatives.

Regulatory Compliance: Compliance with financial regulations is critical. The model should adhere to laws and regulations governing fraud detection and customer data protection.

Feedback Loop: The model should have a feedback loop that incorporates outcomes of past fraud cases to improve its accuracy over time.

---

# Model Integration Guide (Backend Team)

**Audience:** Backend engineers who will load `fraud_detection_model.pkl` and expose it as a service. This section covers installing and calling the model only. How you wrap it (FastAPI, Flask, Django, a queue worker, etc.) is up to you.

## 1. What this model does

Given a single bank payment, it returns the **probability that the payment is fraudulent** and a fraud / not-fraud decision.

- **Algorithm:** Random Forest Classifier (100 trees, `max_depth=8`, `class_weight="balanced"`), wrapped in an sklearn `Pipeline` that also contains the preprocessing (categorical encoding + scaling). You send **raw** transaction fields. No manual encoding is needed.
- **Training data:** `bs140513_032310.csv`, the BankSim synthetic bank payments dataset (594,643 transactions, 1.2% fraud).
- **Source notebook:** `Sourcecode.ipynb`. `train_model.py` reproduces it end to end.

## 2. Input fields

All 7 fields are required. Send values **without** the single quotes that appear in the raw CSV (`C1093826151`, not `'C1093826151'`).

| Field | Type | Meaning | Valid values |
|---|---|---|---|
| `step` | int | Day of the simulation the payment happened on | 0 – 179 in training |
| `customer` | string | Customer ID | e.g. `C1093826151` (unseen IDs are allowed, see §6) |
| `age` | string | Age group | `0`–`6`, or `U` (unknown) |
| `gender` | string | Gender | `E` (enterprise), `F`, `M`, `U` (unknown) |
| `merchant` | string | Merchant ID | e.g. `M348934600` (unseen IDs are allowed, see §6) |
| `category` | string | Merchant category | `es_barsandrestaurants`, `es_contents`, `es_fashion`, `es_food`, `es_health`, `es_home`, `es_hotelservices`, `es_hyper`, `es_leisure`, `es_otherservices`, `es_sportsandtoys`, `es_tech`, `es_transportation`, `es_travel`, `es_wellnessandbeauty` |
| `amount` | float | Payment amount | ≥ 0 (0 – 8,329.96 in training) |

`zipcodeOri` and `zipMerchant` from the CSV are **not** inputs. They have a single value across the whole dataset, so they were dropped.

## 3. Output

| Field | Type | Meaning |
|---|---|---|
| `fraud_probability` | float | 0.0 – 1.0, probability the payment is fraud |
| `is_fraud` | bool | `fraud_probability >= threshold` |
| `threshold` | float | Decision threshold used (default `0.5`) |
| `unknown_ids` | list | `"customer"` and/or `"merchant"` if that ID was never seen in training |
| `model_version` | string | Version of the `.pkl` that produced the result |

## 4. Performance (30% held-out test set, 178,393 payments)

| Metric | Value |
|---|---|
| ROC AUC | 0.996 |
| Fraud recall | 0.98 (catches 2,124 of 2,160 frauds) |
| Fraud precision | 0.23 (about 1 in 4 flagged payments is real fraud) |
| Accuracy | 0.96 |

The model is tuned to **miss almost no fraud** at the cost of false alarms. Treat `is_fraud = True` as "send to review / step-up verification", not "block automatically". To trade recall for precision, compare `fraud_probability` against a higher threshold on your side. The bundle's `decision_threshold` is the default only.

## 5. Installing the model into your project

**Step 1: Copy these 3 files into your backend project** (e.g. into an `ml/fraud/` folder):

```
fraud_detection_model.pkl   # the trained model (required)
predict.py                  # reference loader + prediction function (recommended)
requirements.txt            # exact library versions the model was trained with
```

**Step 2: Install the pinned dependencies.** Use Python 3.11+ (the model was trained on Python 3.14).

```bash
# from inside your backend project, with its virtual environment activated
pip install -r ml/fraud/requirements.txt
```

Or merge these lines into your project's existing `requirements.txt` / `pyproject.toml`:

```
pandas==3.0.2
numpy==2.4.4
scikit-learn==1.9.1
joblib==1.5.3
```

> **Pin `scikit-learn` to exactly `1.9.1`.** A pickled sklearn model is only guaranteed to load in the same sklearn version it was saved with. If you must upgrade sklearn, ask the ML team to re-run `train_model.py` with the new version and send you a fresh `.pkl`.

**Step 3: Sanity-check that the model loads and predicts.**

```bash
cd ml/fraud
python predict.py
```

Expected output:

```
{'step': 0, 'customer': 'C1332295774', 'age': '3', 'gender': 'M', 'merchant': 'M480139044', 'category': 'es_health', 'amount': 324.5}
  -> {'fraud_probability': 0.987, 'is_fraud': True, 'threshold': 0.5, 'unknown_ids': [], 'model_version': '1.0.0'}
{'step': 0, 'customer': 'C1093826151', 'age': '4', 'gender': 'M', 'merchant': 'M348934600', 'category': 'es_transportation', 'amount': 4.55}
  -> {'fraud_probability': 0.0075, 'is_fraud': False, 'threshold': 0.5, 'unknown_ids': [], 'model_version': '1.0.0'}
```

If you see this, the `.pkl` is installed correctly.

## 6. Calling the model from your code

**Option A (recommended): use `predict.py`.** It handles loading, caching, validation and column order.

```python
from predict import load_bundle, predict_transaction

load_bundle()   # call ONCE at startup; loads the .pkl into memory and caches it

result = predict_transaction({
    "step": 0,
    "customer": "C1332295774",
    "age": "3",
    "gender": "M",
    "merchant": "M480139044",
    "category": "es_health",
    "amount": 324.5,
})
# -> {'fraud_probability': 0.987, 'is_fraud': True, 'threshold': 0.5,
#     'unknown_ids': [], 'model_version': '1.0.0'}
```

`predict_transaction` raises `ValueError` for missing fields, an `age`/`gender`/`category` outside the allowed values, or a negative `amount`. Map that to a 400/422 response in your service.

`load_bundle()` reads `fraud_detection_model.pkl` from the **current working directory** by default. If your service runs from another directory, pass the path explicitly: `load_bundle("ml/fraud/fraud_detection_model.pkl")`.

**Option B: load the `.pkl` directly** (if you don't want `predict.py`):

```python
import pickle
import pandas as pd

with open("fraud_detection_model.pkl", "rb") as f:   # once, at startup
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
X = pd.DataFrame([transaction], columns=bundle["feature_names"])  # column order matters
fraud_probability = pipeline.predict_proba(X)[0, 1]
is_fraud = fraud_probability >= bundle["decision_threshold"]
```

`joblib.load("fraud_detection_model.pkl")` also works if your project already uses joblib.

**What's inside the `.pkl`:** a dict with

```python
{
  "pipeline": <fitted sklearn Pipeline: OrdinalEncoder + StandardScaler + RandomForestClassifier>,
  "model_name": "Random Forest Classifier",
  "model_version": "1.0.0",
  "feature_names": ["step", "customer", "age", "gender", "merchant", "category", "amount"],
  "categorical_features": [...], "numerical_features": [...],
  "decision_threshold": 0.5,
  "allowed_values": {"age": [...], "gender": [...], "category": [...]},
  "training_data_range": {"step": [0, 179], "amount": [0.0, 8329.96]},
  "metrics": {...},
  "sklearn_version": "1.9.1", "pandas_version": "3.0.2", "trained_at_utc": "..."
}
```

`model_metadata.json` holds the same content without the pipeline. Read it when you only need the version, metrics or allowed values without loading the model.

## 7. Things to carry into the service

1. **Load once, not per request.** Loading the 5.5 MB pickle takes noticeably longer than a prediction. Do it at process startup.
2. **Security: only load the `.pkl` from this repo.** Unpickling can execute arbitrary code. Never load a `.pkl` that came from user input or an untrusted source.
3. **Unseen customer/merchant IDs** are encoded as "unknown" instead of raising an error, and are reported in `unknown_ids`. The model learned partly from customer/merchant identity, so scores for unknown IDs are less reliable. Consider logging them.
4. **Log `model_version`** with every prediction so you can tell which model produced each decision after a retrain.
5. **Synthetic data caveat:** this model was trained on simulated BankSim data. Validate it against real transactions before relying on it in production.

## 8. Retraining (ML team)

```bash
pip install -r requirements.txt
python train_model.py      # ~40 seconds; overwrites fraud_detection_model.pkl + model_metadata.json
```

Bump `MODEL_VERSION` in `train_model.py` before retraining, then hand the new `.pkl` to the backend team.

## 9. Files in this folder

| File | Purpose |
|---|---|
| `bs140513_032310.csv` | Training data (BankSim synthetic payments) |
| `Sourcecode.ipynb` | Original exploratory notebook |
| `CS-2-Fraud Detection-Predication on Bank Payments.pdf` | Case study write-up |
| `train_model.py` | Reproducible script: CSV → trained pipeline → `fraud_detection_model.pkl` |
| `predict.py` | Reference loader and prediction function for the backend |
| `fraud_detection_model.pkl` | **The model artifact the backend loads** |
| `model_metadata.json` | Human-readable copy of the bundle's metadata (no model inside) |
| `requirements.txt` | Pinned library versions used to train and serve the model |
| `HOW_TO_CREATE_A_PICKLE.md` | Tutorial: how the `.pkl` was created, step by step |
