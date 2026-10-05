"""
Minimal reference implementation for using fraud_detection_model.pkl.

This is what the backend team's handler should mirror: load the bundle once at
process startup, then call predict_transaction(...) per transaction.

Example:
    python predict.py                      # runs two built-in demo transactions
    python predict.py transaction.json     # scores a transaction stored in a JSON file
"""
import json
import pickle
import sys

import pandas as pd

MODEL_PATH = "fraud_detection_model.pkl"

_bundle = None


def load_bundle(path: str = MODEL_PATH) -> dict:
    """Load the pickled bundle once and cache it (never load it per request)."""
    global _bundle
    if _bundle is None:
        # "rb" = read binary. Only ever load .pkl files you trust: unpickling
        # can execute arbitrary code.
        with open(path, "rb") as f:
            _bundle = pickle.load(f)
    return _bundle


def _validate(transaction: dict, bundle: dict) -> None:
    missing = [name for name in bundle["feature_names"] if name not in transaction]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    for field, allowed in bundle["allowed_values"].items():
        if str(transaction[field]) not in allowed:
            raise ValueError(f"{field}={transaction[field]!r} is not one of {allowed}")

    if float(transaction["amount"]) < 0:
        raise ValueError("amount must be >= 0")


def _unknown_ids(transaction: dict, bundle: dict) -> list:
    """Customer/merchant IDs the model never saw in training (scored as 'unknown')."""
    encoder = bundle["pipeline"].named_steps["preprocess"].named_transformers_["categorical"]
    known = dict(zip(bundle["categorical_features"], encoder.categories_))
    return [
        field for field in ("customer", "merchant")
        if str(transaction[field]) not in known[field]
    ]


def predict_transaction(transaction: dict) -> dict:
    """
    Score a single raw transaction.

    `transaction` must contain: step, customer, age, gender, merchant, category,
    amount (values without the surrounding quotes used in the raw CSV).
    """
    bundle = load_bundle()
    _validate(transaction, bundle)

    # DataFrame with the training column names, in the training column order.
    row = {name: transaction[name] for name in bundle["feature_names"]}
    for name in bundle["categorical_features"]:
        row[name] = str(row[name])
    X = pd.DataFrame([row], columns=bundle["feature_names"])

    probability = float(bundle["pipeline"].predict_proba(X)[0, 1])
    threshold = bundle["decision_threshold"]

    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= threshold,
        "threshold": threshold,
        "unknown_ids": _unknown_ids(transaction, bundle),
        "model_version": bundle["model_version"],
    }


DEMO_TRANSACTIONS = [
    # Real fraudulent row from the dataset
    {"step": 0, "customer": "C1332295774", "age": "3", "gender": "M",
     "merchant": "M480139044", "category": "es_health", "amount": 324.5},
    # Real legitimate row from the dataset
    {"step": 0, "customer": "C1093826151", "age": "4", "gender": "M",
     "merchant": "M348934600", "category": "es_transportation", "amount": 4.55},
]


if __name__ == "__main__":
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            transactions = [json.load(f)]
    elif len(sys.argv) == 1:
        transactions = DEMO_TRANSACTIONS
    else:
        print("Usage: python predict.py [transaction.json]")
        sys.exit(1)

    for transaction in transactions:
        print(transaction)
        print("  ->", predict_transaction(transaction))
