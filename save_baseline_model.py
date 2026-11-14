
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------
# 1. Load original artifact information
# ---------------------------------------

original_artifact = joblib.load(
    "models/best_breast_cancer_model.pkl"
)

feature_names = original_artifact["feature_names"]

# ---------------------------------------
# 2. Load dataset
# ---------------------------------------

df = pd.read_csv("data/breast_cancer_wdbc.csv")

target_column = "diagnosis"

if target_column not in df.columns:
    raise ValueError("Diagnosis column not found.")

y = df[target_column]

if y.dtype == "object":
    y = y.astype(str).str.strip().str.upper()
    y = y.map({
        "B": 0,
        "M": 1,
        "BENIGN": 0,
        "MALIGNANT": 1
    })

if y.isna().any():
    raise ValueError("Invalid diagnosis labels found.")

y = y.astype(int)

X = df[feature_names]

# ---------------------------------------
# 3. Recreate the same data split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------------------------------
# 4. Create and train Baseline SVM
# ---------------------------------------

baseline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

baseline_svm.fit(X_train, y_train)

# ---------------------------------------
# 5. Create model artifact
# ---------------------------------------

baseline_artifact = {
    "model": baseline_svm,
    "model_name": "Baseline SVM",
    "threshold": 0.23,
    "feature_names": feature_names,
    "default_values": X_train.median().to_dict()
}

# ---------------------------------------
# 6. Save separately
# ---------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    baseline_artifact,
    "models/baseline_svm_model.pkl"
)

print("Baseline SVM saved successfully!")
print("Model: Baseline SVM")
print("Threshold: 0.23")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Saved location: models/baseline_svm_model.pkl")
