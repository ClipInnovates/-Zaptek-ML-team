"""
Train and export the banking fraud detection model as a pickle file.

Reproduces the modeling pipeline from `Sourcecode.ipynb`:
  1. Load bs140513_032310.csv (BankSim synthetic bank payments)
  2. Strip the quote characters around string values ('C1093826151' -> C1093826151)
  3. Drop zipcodeOri and zipMerchant (each has a single unique value)
  4. Encode categorical columns as integer codes, scale step/amount
  5. Train a Random Forest (100 trees, max_depth=8, class_weight="balanced")
     on a stratified 70/30 split
  6. Save the fitted pipeline + metadata as fraud_detection_model.pkl

Unlike the notebook (which encoded with `.cat.codes` on the full dataset), the
encoding and scaling are fitted inside an sklearn Pipeline. That way the exact
same preprocessing is stored in the .pkl and re-applied at prediction time, so
the backend can send raw transaction fields instead of pre-encoded numbers.

Run:
    python train_model.py
Output:
    fraud_detection_model.pkl
    model_metadata.json
"""
import json
import pickle
from datetime import datetime, timezone

import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

CSV_PATH = "bs140513_032310.csv"
OUTPUT_PATH = "fraud_detection_model.pkl"
METADATA_PATH = "model_metadata.json"
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5

CATEGORICAL_FEATURES = ["customer", "age", "gender", "merchant", "category"]
NUMERICAL_FEATURES = ["step", "amount"]
FEATURE_NAMES = ["step", "customer", "age", "gender", "merchant", "category", "amount"]
TARGET = "fraud"


def load_data(path: str) -> pd.DataFrame:
    # Single-valued columns carry no information (see notebook cell 13).
    data = pd.read_csv(path).drop(columns=["zipcodeOri", "zipMerchant"])
    # Raw CSV wraps every string value in single quotes; the API should not need them.
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].str.strip("'")
    return data


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [
            # Unknown values at prediction time (e.g. a new customer ID) map to -1
            # instead of raising, so the service never crashes on unseen IDs.
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def main():
    data = load_data(CSV_PATH)
    print(f"Loaded {CSV_PATH}: {data.shape}")

    X = data[FEATURE_NAMES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, shuffle=True, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DECISION_THRESHOLD).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", cm)
    print(f"ROC AUC: {roc_auc:.4f}")

    bundle = {
        "pipeline": pipeline,
        "model_name": "Random Forest Classifier",
        "model_version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "decision_threshold": DECISION_THRESHOLD,
        "allowed_values": {
            "age": sorted(data["age"].unique().tolist()),
            "gender": sorted(data["gender"].unique().tolist()),
            "category": sorted(data["category"].unique().tolist()),
        },
        "training_data_range": {
            "step": [int(data["step"].min()), int(data["step"].max())],
            "amount": [float(data["amount"].min()), float(data["amount"].max())],
        },
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "fraud_precision": round(report["1"]["precision"], 4),
            "fraud_recall": round(report["1"]["recall"], 4),
            "fraud_f1": round(report["1"]["f1-score"], 4),
            "accuracy": round(report["accuracy"], 4),
            "confusion_matrix": cm.tolist(),
            "test_size": int(len(y_test)),
        },
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    # "wb" = write binary: pickle produces bytes, not text.
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved model bundle to {OUTPUT_PATH}")

    # Human-readable sidecar for quick inspection without unpickling
    with open(METADATA_PATH, "w") as f:
        json.dump({k: v for k, v in bundle.items() if k != "pipeline"}, f, indent=2)
    print(f"Saved {METADATA_PATH}")


if __name__ == "__main__":
    main()
