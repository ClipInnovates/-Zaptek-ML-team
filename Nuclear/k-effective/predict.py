"""
Minimal reference implementation for serving reactor_k_eff_model.pkl.

This is what the backend team's API handler should mirror: load the bundle
once at process startup, then call predict_reactor_behavior(...) per request.

Example:
    python predict.py 3.8
    python predict.py 4.2 10.4 1.0
"""
import sys

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "reactor_k_eff_model.pkl"

_bundle = None


def load_bundle(path: str = MODEL_PATH):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict_reactor_behavior(enrichment_percent: float, fuel_density: float, moderator_density: float) -> dict:
    """
    Predict k_effective and reactor status for a given fuel configuration.

    All three inputs are required by the deployed model; the notebook's
    original convenience of "look up the nearest training row's densities"
    is a training-time shortcut, not something a backend should replicate,
    since it silently fabricates two of the three inputs.
    """
    bundle = load_bundle()
    lo, hi = bundle["training_data_range"]["enrichment_percent"]
    if not (lo <= enrichment_percent <= hi):
        raise ValueError(
            f"enrichment_percent={enrichment_percent} is outside the training "
            f"range [{lo}, {hi}]; predictions outside this range are extrapolation."
        )

    # Pass a DataFrame with the training column names to avoid sklearn's
    # "X does not have valid feature names" warning and keep column order explicit.
    X = pd.DataFrame(
        [[enrichment_percent, fuel_density, moderator_density]],
        columns=bundle["feature_names"],
    )

    k_eff_pred = float(bundle["regressor"].predict(X)[0])
    status_pred = str(bundle["classifier"].predict(X)[0])

    return {
        "enrichment_percent": enrichment_percent,
        "fuel_density": fuel_density,
        "moderator_density": moderator_density,
        "predicted_k_eff": k_eff_pred,
        "predicted_status": status_pred,
        "model": bundle["regressor_name"],
    }


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]]
    if len(args) == 1:
        enrichment = args[0]
        # Default mid-range densities when only enrichment is given (demo only).
        fuel_density, moderator_density = 10.3, 1.0
    elif len(args) == 3:
        enrichment, fuel_density, moderator_density = args
    else:
        print("Usage: python predict.py <enrichment_percent> [fuel_density moderator_density]")
        sys.exit(1)

    result = predict_reactor_behavior(enrichment, fuel_density, moderator_density)
    print(result)
