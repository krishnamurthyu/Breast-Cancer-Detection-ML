
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. LOAD BASELINE SVM MODEL
# ==========================================

artifact = joblib.load(
    "models/baseline_svm_model.pkl"
)

model = artifact["model"]
model_name = artifact["model_name"]
threshold = float(artifact["threshold"])
feature_names = artifact["feature_names"]

print("Model:", model_name)
print("Threshold:", threshold)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "data/breast_cancer_wdbc.csv"
)

y = df["diagnosis"]

if y.dtype == "object":
    y = (
        y.astype(str)
        .str.strip()
        .str.upper()
        .map({
            "B": 0,
            "M": 1,
            "BENIGN": 0,
            "MALIGNANT": 1
        })
    )

if y.isna().any():
    raise ValueError("Invalid diagnosis labels found.")

y = y.astype(int)

X = df[feature_names]

# ==========================================
# 3. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 4. PREDICTIONS
# ==========================================

probabilities = model.predict_proba(X_test)

malignant_index = list(model.classes_).index(1)

malignant_probabilities = probabilities[:, malignant_index]

predictions = (
    malignant_probabilities >= threshold
).astype(int)

# ==========================================
# 5. CALCULATE METRICS
# ==========================================

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

f2 = fbeta_score(
    y_test, predictions, beta=2, zero_division=0
)

auc = roc_auc_score(
    y_test, malignant_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n========== FINAL MODEL PERFORMANCE ==========")

print("Model:", model_name)
print("Threshold:", threshold)
print("Test records:", len(X_test))

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"F2 Score: {f2:.4f}")
print(f"AUC-ROC: {auc:.4f}")

print("\n========== CONFUSION MATRIX VALUES ==========")

print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        zero_division=0
    )
)

# ==========================================
# 7. CREATE REPORTS FOLDER
# ==========================================

os.makedirs("reports", exist_ok=True)

# ==========================================
# 8. CONFUSION MATRIX PLOT
# ==========================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Benign", "Malignant"]
)

display.plot(values_format="d")

plt.title("Baseline SVM - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "reports/baseline_svm_confusion_matrix.png",
    dpi=300
)

plt.close()

# ==========================================
# 9. ROC CURVE
# ==========================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    malignant_probabilities
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Baseline SVM (AUC = {auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("Baseline SVM - ROC Curve")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "reports/baseline_svm_roc_curve.png",
    dpi=300
)

plt.close()

# ==========================================
# 10. SAVE METRICS TO CSV
# ==========================================

metrics = pd.DataFrame([{
    "Model": model_name,
    "Threshold": threshold,
    "Accuracy (%)": round(accuracy * 100, 2),
    "Precision (%)": round(precision * 100, 2),
    "Recall (%)": round(recall * 100, 2),
    "F1 Score": round(f1, 4),
    "F2 Score": round(f2, 4),
    "AUC-ROC": round(auc, 4),
    "True Negatives": tn,
    "False Positives": fp,
    "False Negatives": fn,
    "True Positives": tp
}])

metrics.to_csv(
    "reports/baseline_svm_metrics.csv",
    index=False
)

# ==========================================
# 11. SAVE CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    zero_division=0
)

with open(
    "reports/baseline_svm_classification_report.txt",
    "w"
) as file:

    file.write("BASELINE SVM - FINAL EVALUATION\n\n")
    file.write(f"Threshold: {threshold}\n\n")
    file.write(report)

print("\n========== REPORTS SAVED ==========")

print("1. baseline_svm_confusion_matrix.png")
print("2. baseline_svm_roc_curve.png")
print("3. baseline_svm_metrics.csv")
print("4. baseline_svm_classification_report.txt")

print("\nAll reports generated successfully!")
