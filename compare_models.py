
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/breast_cancer_wdbc.csv")

target_candidates = ["diagnosis", "Diagnosis", "target", "label"]

target_column = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_column is None:
    raise ValueError("Diagnosis column not found in dataset.")

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

# Exclude diagnosis and any ID column from features
X = df.drop(columns=[target_column])

id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "id_number", "patient_id"]
]

X = X.drop(columns=id_columns)

# Keep numeric features only
X = X.select_dtypes(include=[np.number])

print("Dataset shape:", X.shape)
print("Benign records:", (y == 0).sum())
print("Malignant records:", (y == 1).sum())

# --------------------------------------------------
# 2. Same train/test split
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
# 3. Define three baseline models
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}

# --------------------------------------------------
# 4. Cross-validation setup
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# --------------------------------------------------
# 5. Train, select threshold and evaluate
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 55)
    print("MODEL:", model_name)
    print("=" * 55)

    # Out-of-fold probabilities from training data
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )

    malignant_index = list(np.unique(y_train)).index(1)

    oof_malignant_prob = oof_probabilities[:, malignant_index]

    # Select threshold using training OOF predictions
    threshold_scores = []

    for threshold in np.arange(0.05, 0.951, 0.01):

        oof_predictions = (
            oof_malignant_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_train,
            oof_predictions,
            zero_division=0
        )

        f2 = fbeta_score(
            y_train,
            oof_predictions,
            beta=2,
            zero_division=0
        )

        threshold_scores.append({
            "threshold": threshold,
            "f2": f2,
            "precision": precision
        })

    best = max(
        threshold_scores,
        key=lambda item: (
            item["f2"],
            item["precision"],
            item["threshold"]
        )
    )

    selected_threshold = best["threshold"]

    print("Selected threshold:", round(selected_threshold, 4))
    print("Training OOF F2:", round(best["f2"], 4))

    # Fit model on all training data
    model.fit(X_train, y_train)

    # Predict test probabilities
    test_probabilities = model.predict_proba(X_test)

    test_malignant_prob = test_probabilities[:, malignant_index]

    test_predictions = (
        test_malignant_prob >= selected_threshold
    ).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_test, test_predictions)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f2 = fbeta_score(
        y_test,
        test_predictions,
        beta=2,
        zero_division=0
    )

    auc = roc_auc_score(y_test, test_malignant_prob)

    print("Test accuracy:", round(accuracy * 100, 2), "%")
    print("Test precision:", round(precision * 100, 2), "%")
    print("Malignant recall:", round(recall * 100, 2), "%")
    print("Test F2-score:", round(f2, 4))
    print("AUC-ROC:", round(auc, 4))
    print("True Negatives:", tn)
    print("False Positives:", fp)
    print("False Negatives:", fn)
    print("True Positives:", tp)

    results.append({
        "Model": model_name,
        "Threshold": round(selected_threshold, 4),
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
# 6. Final comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

# --------------------------------------------------
# 7. Save comparison report
# --------------------------------------------------

os.makedirs("reports", exist_ok=True)

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved to reports/model_comparison.csv")
