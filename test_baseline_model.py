
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score
)

# Load baseline model
artifact = joblib.load(
    "models/baseline_svm_model.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]
feature_names = artifact["feature_names"]

# Load dataset
df = pd.read_csv("data/breast_cancer_wdbc.csv")

y = df["diagnosis"]

if y.dtype == "object":
    y = y.astype(str).str.strip().str.upper()
    y = y.map({"B": 0, "M": 1})

y = y.astype(int)
X = df[feature_names]

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict
probabilities = model.predict_proba(X_test)

malignant_index = list(model.classes_).index(1)

malignant_probabilities = probabilities[:, malignant_index]

predictions = (
    malignant_probabilities >= threshold
).astype(int)

# Evaluate
tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
).ravel()

print("Model:", artifact["model_name"])
print("Threshold:", threshold)

print("\nAccuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("Precision:",
      round(precision_score(y_test, predictions) * 100, 2), "%")

print("Malignant Recall:",
      round(recall_score(y_test, predictions) * 100, 2), "%")

print("F2 Score:",
      round(fbeta_score(y_test, predictions, beta=2), 4))

print("AUC-ROC:",
      round(roc_auc_score(y_test, malignant_probabilities), 4))

print("\nConfusion Matrix:")
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)
