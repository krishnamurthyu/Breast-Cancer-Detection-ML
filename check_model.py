
import joblib
import pandas as pd
import os

# File paths
model_path = "models/best_breast_cancer_model.pkl"
data_path = "data/breast_cancer_wdbc.csv"

print("===== FILE VERIFICATION =====")

print("Model exists:", os.path.exists(model_path))
print("Dataset exists:", os.path.exists(data_path))

# Load model
artifact = joblib.load(model_path)

# Load dataset
df = pd.read_csv(data_path)

print("\n===== MODEL INFORMATION =====")

print("Selected model:", artifact["model_name"])
print("Threshold:", artifact["threshold"])
print("Number of features:", len(artifact["feature_names"]))

print("\n===== DATASET INFORMATION =====")

print("Dataset shape:", df.shape)

print("Column names:")
print(df.columns.tolist())

print("\nModel and dataset loaded successfully!")
