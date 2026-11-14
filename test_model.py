import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load model and dataset
# --------------------------------------------------

artifact = joblib.load("models/best_breast_cancer_model.pkl")

model = artifact["model"]
model_name = artifact["model_name"]
current_threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

df = pd.read_csv("data/breast_cancer_wdbc.csv")

print("Model:", model_name)
print("Current threshold:", round(current_threshold, 4))

# --------------------------------------------------
# 2. Identify diagnosis column
# --------------------------------------------------

possible_targets = ["diagnosis", "Diagnosis", "target", "label"]

target_column = next(
    (col for col in possible_targets if col in df.columns),
    None
)

if target_column is None:
    raise ValueError("Diagnosis column not found in the CSV.")

# Convert diagnosis into numeric labels:
# Benign = 0, Malignant = 1

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
    raise ValueError("Diagnosis column contains unknown or missing labels.")

y = y.astype(int)

X = df[feature_names]

# --------------------------------------------------
# 3. Recreate the same train/test split
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
# 4. Get malignant probabilities
# --------------------------------------------------

probabilities = model.predict_proba(X_test)

malignant_class_index = list(model.classes_).index(1)

malignant_probabilities = probabilities[:, malignant_class_index]

# --------------------------------------------------
# 5. Compare different thresholds
# --------------------------------------------------

thresholds = [
    current_threshold,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40
]

results = []

print("\n========== THRESHOLD COMPARISON ==========")

for threshold in thresholds:

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
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    results.append({
        "Threshold": round(threshold, 4),
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Malignant Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2),
        "False Negatives": fn,
        "False Positives": fp,
        "True Positives": tp,
        "True Negatives": tn
    })

    print(f"\nThreshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Malignant Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("False Negatives:", fn)
    print("False Positives:", fp)

# --------------------------------------------------
# 6. Display comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== FINAL COMPARISON TABLE ==========")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. Identify false negatives at current threshold
# --------------------------------------------------

current_predictions = (
    malignant_probabilities >= current_threshold
).astype(int)

false_negative_mask = (
    (y_test.to_numpy() == 1) &
    (current_predictions == 0)
)

false_negative_records = X_test.loc[false_negative_mask].copy()

false_negative_records["Actual Diagnosis"] = "Malignant"
false_negative_records["Malignant Probability (%)"] = (
    malignant_probabilities[false_negative_mask] * 100
).round(2)

print("\n========== FALSE NEGATIVE RECORDS ==========")
print("Total false negatives:", false_negative_mask.sum())

print(
    false_negative_records[
        ["Actual Diagnosis", "Malignant Probability (%)"]
    ].to_string()
)

# --------------------------------------------------
# 8. AUC-ROC (independent of classification threshold)
# --------------------------------------------------

auc = roc_auc_score(y_test, malignant_probabilities)

print("\nAUC-ROC:", round(auc, 4))

# --------------------------------------------------
# 9. Save comparison report
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/threshold_comparison.csv",
    index=False
)

print("\nReport saved successfully!")

print("\nReport saved to reports/threshold_comparison.csv")
