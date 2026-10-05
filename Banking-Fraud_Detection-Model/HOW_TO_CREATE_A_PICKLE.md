# How to Create (and Load) a Pickle File for an ML Model

This walks through exactly how `fraud_detection_model.pkl` was made, so you can do the same for any model.

## What a pickle file is

Your trained model only exists in RAM while your notebook is running. Close the notebook and it's gone.
**Pickling** turns a Python object (a model, a dict, a list…) into bytes saved on disk. **Unpickling** turns those bytes back into the exact same object in another program, like the backend service, without retraining.

```
trained model in memory  --pickle.dump-->  model.pkl (bytes on disk)  --pickle.load-->  same model in another program
```

## Step 0: The smallest possible example

```python
import pickle

data = {"name": "ZapTek", "scores": [1, 2, 3]}

# SAVE: "wb" = write binary
with open("example.pkl", "wb") as f:
    pickle.dump(data, f)

# LOAD: "rb" = read binary
with open("example.pkl", "rb") as f:
    loaded = pickle.load(f)

print(loaded)   # {'name': 'ZapTek', 'scores': [1, 2, 3]}
```

That's all pickle is: `dump` to save, `load` to read back. Always use **binary** mode (`"wb"` / `"rb"`).

## Step 1: Put ALL preprocessing inside the model (most important step)

In the notebook, the data was transformed *before* training:

```python
data_reduced[col_categorical].apply(lambda x: x.cat.codes)        # 'es_health' -> 4
X_train[numerical_columns] = scaler.fit_transform(X_train[...])   # scale amount/step
rf_clf.fit(X_train, y_train)
```

If you pickle only `rf_clf`, the backend gets a model that expects **numbers**, like `category=4`. But they'll receive `"es_health"`, and they don't have your encoding or your scaler. They can't use the model correctly.

**Fix:** use an sklearn `Pipeline` so the encoder, scaler and model are a single object:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

preprocess = ColumnTransformer([
    ("categorical",
     OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),  # don't crash on new IDs
     ["customer", "age", "gender", "merchant", "category"]),
    ("numerical", StandardScaler(), ["step", "amount"]),
])

pipeline = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8,
                                          class_weight="balanced", random_state=42)),
])
```

Rule of thumb: **whatever you do to the data before `.fit()`, the backend must do before `.predict()`.** Put it in the pipeline and it travels inside the `.pkl` automatically.

## Step 2: Train it

```python
pipeline.fit(X_train, y_train)            # X_train has RAW values: "es_health", "M", 324.5 ...
print(pipeline.predict_proba(X_test)[:5]) # check it works BEFORE saving
```

## Step 3: Bundle the model with its metadata

The model alone doesn't tell the backend which columns it needs, in what order, or which sklearn version made it. Save a dict:

```python
import sklearn

bundle = {
    "pipeline": pipeline,
    "model_version": "1.0.0",
    "feature_names": ["step", "customer", "age", "gender", "merchant", "category", "amount"],
    "decision_threshold": 0.5,
    "metrics": {"roc_auc": 0.9961, "fraud_recall": 0.9833},
    "sklearn_version": sklearn.__version__,   # backend MUST install this exact version
}
```

## Step 4: Dump it to a .pkl

```python
import pickle

with open("fraud_detection_model.pkl", "wb") as f:
    pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
```

`protocol=pickle.HIGHEST_PROTOCOL` uses the newest, most efficient format.

## Step 5: Test loading it in a FRESH Python process

Don't test in the same notebook, because the model is still in memory there and the test proves nothing. Open a new terminal:

```python
import pickle
import pandas as pd

with open("fraud_detection_model.pkl", "rb") as f:
    bundle = pickle.load(f)

tx = {"step": 0, "customer": "C1332295774", "age": "3", "gender": "M",
      "merchant": "M480139044", "category": "es_health", "amount": 324.5}

X = pd.DataFrame([tx], columns=bundle["feature_names"])
print(bundle["pipeline"].predict_proba(X)[0, 1])   # ~0.987 -> fraud
```

If that prints a sensible probability, the pickle is ready to hand over.

## Step 6: Pin library versions

```bash
pip freeze | findstr /i "scikit-learn pandas numpy joblib"     # Windows
pip freeze | grep -iE "scikit-learn|pandas|numpy|joblib"       # macOS/Linux
```

Put the output in `requirements.txt`. A model pickled with scikit-learn 1.9.1 may fail or behave differently on another version.

## pickle vs joblib

| | `pickle` | `joblib` |
|---|---|---|
| Comes with Python | Yes | Installed with scikit-learn |
| Save | `pickle.dump(obj, f)` | `joblib.dump(obj, "model.pkl")` |
| Load | `pickle.load(f)` | `joblib.load("model.pkl")` |
| Good for | Anything | Large NumPy-heavy models (faster, supports `compress=3`) |

Both are fine for sklearn models. `joblib.load` can read a file written by `pickle.dump`, but not always the other way round. That's why this project uses plain `pickle`: the backend can load it either way. (The `Nuclear/k-effective` model in this repo uses joblib.)

## Common mistakes

| Mistake | What happens | Fix |
|---|---|---|
| Pickling only the classifier, not the preprocessing | Backend predictions are garbage or crash | Use a `Pipeline` (Step 1) |
| Using `lambda` functions in the pipeline | `PicklingError: Can't pickle <lambda>` | Use built-in sklearn transformers or named top-level functions |
| Opening the file in text mode (`"w"` / `"r"`) | `TypeError` / corrupted file | Always `"wb"` / `"rb"` |
| Different sklearn version on the server | `InconsistentVersionWarning` or load errors | Pin versions (Step 6) |
| Column order differs at prediction time | Silently wrong predictions | Build the DataFrame with `columns=bundle["feature_names"]` |
| Loading a `.pkl` from an untrusted source | It can run arbitrary code on your machine | Only load pickles you or your team created |

## The full script

`train_model.py` in this folder does all of the above end to end. Read it top to bottom, then run `python train_model.py` to regenerate the `.pkl` yourself.
