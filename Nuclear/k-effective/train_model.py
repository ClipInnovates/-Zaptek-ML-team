"""
Train and export the k-effective prediction model bundle.

Reproduces the modeling pipeline from `ML _ AI (Nuclear Team).ipynb`:
  1. Load Reactor_Feasibility_Plot.csv
  2. Augment the data via 1D interpolation (denser enrichment grid)
  3. Train regressors (Linear, Decision Tree, Random Forest, Polynomial deg=3)
     to predict k_eff from [enrichment_percent, fuel_density, moderator_density]
  4. Pick the best regressor by test R^2
  5. Train a Decision Tree classifier to predict reaction_status
     (Subcritical / Critical / Supercritical)
  6. Save both models + metadata as a single joblib bundle for backend use

Run:
    python train_model.py
Output:
    reactor_k_eff_model.pkl
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

FEATURE_NAMES = ["enrichment_percent", "fuel_density", "moderator_density"]
CSV_PATH = "Reactor_Feasibility_Plot.csv"
OUTPUT_PATH = "reactor_k_eff_model.pkl"
RANDOM_STATE = 42

# k_eff -> status boundaries (must match reaction_status labeling used across the notebooks)
SUBCRITICAL_MAX = 0.95
SUPERCRITICAL_MIN = 1.05


def classify_reactor_status(k_eff: float) -> str:
    if k_eff < SUBCRITICAL_MAX:
        return "Subcritical"
    elif k_eff <= SUPERCRITICAL_MIN:
        return "Critical"
    else:
        return "Supercritical"


def augment_data(original_data: pd.DataFrame, num_interpolated_points: int = 300) -> pd.DataFrame:
    """Densify the enrichment grid via linear interpolation, mirroring the notebook."""
    sorted_data = original_data.sort_values("enrichment_percent")

    enrichment_fine = np.linspace(
        sorted_data["enrichment_percent"].min(),
        sorted_data["enrichment_percent"].max(),
        num_interpolated_points,
    )

    fuel_density_fine = np.interp(
        enrichment_fine, sorted_data["enrichment_percent"], sorted_data["fuel_density"]
    )
    moderator_density_fine = np.interp(
        enrichment_fine, sorted_data["enrichment_percent"], sorted_data["moderator_density"]
    )
    k_eff_fine = np.interp(
        enrichment_fine, sorted_data["enrichment_percent"], sorted_data["k_eff"]
    )

    augmented = pd.DataFrame(
        {
            "enrichment_percent": enrichment_fine,
            "fuel_density": fuel_density_fine,
            "moderator_density": moderator_density_fine,
            "k_eff": k_eff_fine,
        }
    )
    augmented["reaction_status"] = pd.cut(
        augmented["k_eff"],
        bins=[-np.inf, SUBCRITICAL_MAX, SUPERCRITICAL_MIN, np.inf],
        labels=["Subcritical", "Critical", "Supercritical"],
    )
    return augmented


def main():
    data = pd.read_csv(CSV_PATH)
    print(f"Loaded {CSV_PATH}: {data.shape}")

    augmented = augment_data(data, num_interpolated_points=300)
    combined = pd.concat([data, augmented], ignore_index=True)
    combined = combined.drop_duplicates(subset=["enrichment_percent"]).sort_values(
        "enrichment_percent"
    )
    print(f"Combined training data: {combined.shape}")

    # ---- Regression: predict k_eff ----
    X = combined[FEATURE_NAMES]
    y = combined["k_eff"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidate_models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=100),
        "Polynomial Regression (deg=3)": Pipeline(
            [("poly", PolynomialFeatures(degree=3)), ("linear", LinearRegression())]
        ),
    }

    reg_results = {}
    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        reg_results[name] = {
            "model": model,
            "r2": r2_score(y_test, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        }
        print(f"{name}: R2={reg_results[name]['r2']:.4f} RMSE={reg_results[name]['rmse']:.4f}")

    best_name = max(reg_results, key=lambda k: reg_results[k]["r2"])
    best_model = reg_results[best_name]["model"]
    print(f"\nBest regressor: {best_name}")

    # ---- Classification: predict reaction_status ----
    combined["status_class"] = combined["k_eff"].apply(classify_reactor_status)
    X_clf = combined[FEATURE_NAMES]
    y_clf = combined["status_class"]

    # Stratification needs >=2 samples per class; the smoothed, interpolated
    # data can leave a boundary class (e.g. Supercritical) with a single row,
    # so fall back to a plain split when that happens instead of crashing.
    class_counts = y_clf.value_counts()
    can_stratify = (class_counts >= 2).all()
    if not can_stratify:
        print(
            f"Warning: class counts too small to stratify ({class_counts.to_dict()}); "
            "using a non-stratified split."
        )
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf,
        y_clf,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_clf if can_stratify else None,
    )
    clf_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=4)
    clf_model.fit(X_train_clf, y_train_clf)
    clf_accuracy = accuracy_score(y_test_clf, clf_model.predict(X_test_clf))
    print(f"Classifier accuracy: {clf_accuracy:.4f}")

    # ---- Bundle everything the backend needs into one artifact ----
    bundle = {
        "regressor": best_model,
        "regressor_name": best_name,
        "classifier": clf_model,
        "feature_names": FEATURE_NAMES,
        "status_thresholds": {
            "subcritical_max": SUBCRITICAL_MAX,
            "supercritical_min": SUPERCRITICAL_MIN,
        },
        "metrics": {
            "regressor_r2": reg_results[best_name]["r2"],
            "regressor_rmse": reg_results[best_name]["rmse"],
            "classifier_accuracy": float(clf_accuracy),
        },
        "training_data_range": {
            "enrichment_percent": [float(data["enrichment_percent"].min()), float(data["enrichment_percent"].max())],
            "fuel_density": [float(data["fuel_density"].min()), float(data["fuel_density"].max())],
            "moderator_density": [float(data["moderator_density"].min()), float(data["moderator_density"].max())],
        },
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(bundle, OUTPUT_PATH)
    print(f"\nSaved model bundle to {OUTPUT_PATH}")

    # Human-readable sidecar for quick inspection without unpickling
    with open("model_metadata.json", "w") as f:
        json.dump(
            {k: v for k, v in bundle.items() if k not in ("regressor", "classifier")},
            f,
            indent=2,
        )
    print("Saved model_metadata.json")


if __name__ == "__main__":
    main()
