
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load original tuned SVM artifact
# --------------------------------------------------

artifact = joblib.load(
    "models/best_breast_cancer_model.pkl"
)

original_svm = artifact["model"]
feature_names = artifact["feature_names"]
original_threshold = float(artifact["threshold"])

print("Original model:", artifact["model_name"])
print("Original threshold:", round(original_threshold, 4))

# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

df = pd.read_csv("data/breast_cancer_wdbc.csv")

target_candidates = [
    "diagnosis", "Diagnosis", "target", "label"
]

target_column = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_column is None:
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
    raise ValueError("Unknown or missing diagnosis labels.")

y = y.astype(int)

X = df[feature_names]

# --------------------------------------------------
# 3. Same train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# --------------------------------------------------
# 4. Define baseline models
# --------------------------------------------------

baseline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ))
])

logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=5000))
])

# --------------------------------------------------
# 5. Prepare comparison models
# --------------------------------------------------

models = [
    (
        "Original Tuned SVM",
        original_svm,
        original_threshold,
        False
    ),
    (
        "Baseline SVM",
        baseline_svm,
        0.23,
        True
    ),
    (
        "Logistic Regression",
        logistic_regression,
        0.18,
        True
    )
]

results = []

# --------------------------------------------------
# 6. Fit and evaluate each model
# --------------------------------------------------

for name, model, threshold, needs_fit in models:

    print("\n" + "=" * 55)
    print("MODEL:", name)
    print("=" * 55)

    if needs_fit:
        model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)

    malignant_index = list(model.classes_).index(1)

    malignant_probabilities = probabilities[:, malignant_index]

    predictions = (
        malignant_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )

    recall = recall_score(
        y_test, predictions, zero_division=0
    )

    f2 = fbeta_score(
        y_test, predictions, beta=2, zero_division=0
    )

    auc = roc_auc_score(
        y_test, malignant_probabilities
    )

    print("Threshold:", round(threshold, 4))
    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Malignant Recall:", round(recall * 100, 2), "%")
    print("F2 Score:", round(f2, 4))
    print("AUC-ROC:", round(auc, 4))
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    results.append({
        "Model": name,
        "Threshold": round(threshold, 4),
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Malignant Recall (%)": round(recall * 100, 2),
        "F2 Score": round(f2, 4),
        "AUC-ROC": round(auc, 4),
        "False Negatives": fn,
        "False Positives": fp,
        "True Positives": tp,
        "True Negatives": tn
    })

# --------------------------------------------------
# 7. Final comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== FINAL THREE-MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 8. Save report
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/final_model_comparison.csv",
    index=False
)

print("\nFinal comparison saved successfully.")
