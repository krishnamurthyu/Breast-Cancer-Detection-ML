
import pandas as pd
import numpy as np
import joblib

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load model and dataset
# --------------------------------------------------

artifact = joblib.load("models/best_breast_cancer_model.pkl")

model = artifact["model"]
feature_names = artifact["feature_names"]
old_threshold = artifact["threshold"]

df = pd.read_csv("data/breast_cancer_wdbc.csv")

print("Model:", artifact["model_name"])
print("Previous threshold:", round(old_threshold, 4))

# --------------------------------------------------
# 2. Prepare X and y
# --------------------------------------------------

target_candidates = ["diagnosis", "Diagnosis", "target", "label"]

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
# 4. Generate out-of-fold probabilities
#    Training data only
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model_for_cv = clone(model)

oof_probabilities = cross_val_predict(
    model_for_cv,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

malignant_index = list(model.classes_).index(1)

oof_malignant_prob = oof_probabilities[:, malignant_index]

# --------------------------------------------------
# 5. Select threshold using F2-score
# --------------------------------------------------

threshold_results = []

for threshold in np.arange(0.05, 0.951, 0.01):

    predictions = (
        oof_malignant_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_train,
        predictions,
        beta=2,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision,
        "Recall": recall,
        "F2 Score": f2
    })

threshold_df = pd.DataFrame(threshold_results)

# Highest F2; if tied, prefer higher precision
best_row = threshold_df.sort_values(
    by=["F2 Score", "Precision", "Threshold"],
    ascending=[False, False, False]
).iloc[0]

selected_threshold = float(best_row["Threshold"])

print("\n========== THRESHOLD SELECTION ==========")
print("Selected threshold:", round(selected_threshold, 4))
print("Training OOF precision:",
      round(best_row["Precision"] * 100, 2), "%")
print("Training OOF malignant recall:",
      round(best_row["Recall"] * 100, 2), "%")
print("Training OOF F2 score:",
      round(best_row["F2 Score"], 4))

# --------------------------------------------------
# 6. Compare selected threshold on test data
# --------------------------------------------------

test_probabilities = model.predict_proba(X_test)

test_malignant_prob = test_probabilities[:, malignant_index]

test_predictions = (
    test_malignant_prob >= selected_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

print("\n========== TEST SET COMPARISON ==========")
print("Selected threshold:", round(selected_threshold, 4))
print("Accuracy:",
      round(accuracy_score(y_test, test_predictions) * 100, 2), "%")
print("Precision:",
      round(precision_score(y_test, test_predictions,
                            zero_division=0) * 100, 2), "%")
print("Malignant recall:",
      round(recall_score(y_test, test_predictions,
                         zero_division=0) * 100, 2), "%")
print("F2 score:",
      round(fbeta_score(y_test, test_predictions,
                        beta=2, zero_division=0), 4))
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nAUC-ROC:",
      round(roc_auc_score(y_test, test_malignant_prob), 4))

print("\nPrevious threshold:", round(old_threshold, 4))
print("New suggested threshold:", round(selected_threshold, 4))

# --------------------------------------------------
# 7. Save threshold selection report
# --------------------------------------------------

import os

os.makedirs("reports", exist_ok=True)

threshold_df.to_csv(
    "reports/cross_validation_thresholds.csv",
    index=False
)

print("\nCross-validation threshold report saved.")
print("IMPORTANT: The saved model artifact has NOT been changed.")
