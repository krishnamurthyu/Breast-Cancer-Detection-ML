
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# 2. MODEL SELECTION
# ==========================================


# ==========================================
# PROFESSIONAL DASHBOARD HEADER
# ==========================================

st.markdown(
    """
    <h1 style="color:#1E3A8A; margin-bottom:5px;">
        🩺 Breast Cancer Detection System
    </h1>
    <p style="color:#64748B; font-size:17px;">
        Machine Learning Based Benign and Malignant Classification
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# DASHBOARD SUMMARY CARDS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Total Dataset Records",
        value="569",
        help="Total records in the WDBC dataset"
    )

with col2:
    st.metric(
        label="Input Features",
        value="30",
        help="Numerical tumour measurement features"
    )

with col3:
    st.metric(
        label="Available ML Models",
        value="3",
        help="Logistic Regression, Original SVM and Baseline SVM"
    )

st.divider()


model_options = {
    "Tuned Logistic Regression": "models/final_breast_cancer_model.pkl",

    "Original Tuned SVM": "models/best_breast_cancer_model.pkl",

    "Baseline SVM": "models/baseline_svm_model.pkl"
}

# ==========================================
# MODEL SELECTION DASHBOARD
# ==========================================

st.subheader("⚙️ Model Configuration")

model_descriptions = {
    "Tuned Logistic Regression":
        "A tuned classification model that estimates the probability of Benign and Malignant cases.",

    "Original Tuned SVM":
        "An RBF-based Support Vector Machine with tuned hyperparameters.",

    "Baseline SVM":
        "A baseline Support Vector Machine using an RBF kernel."
}

col1, col2 = st.columns([2, 1])

with col1:
    selected_model = st.selectbox(
        "Select Machine Learning Model:",
        options=list(model_options.keys())
    )

with col2:
    st.info("3 trained model options are available.")

st.caption(model_descriptions[selected_model])

# ==========================================
# 3. LOAD SELECTED MODEL AND DATASET
# ==========================================

@st.cache_resource
def load_artifact(model_path):

    return joblib.load(model_path)


@st.cache_data
def load_dataset():

    return pd.read_csv(
        "data/breast_cancer_wdbc.csv"
    )


artifact = load_artifact(
    model_options[selected_model]
)

df = load_dataset()

model = artifact["model"]
model_name = artifact["model_name"]
threshold = float(artifact["threshold"])

feature_names = artifact["feature_names"]
default_values = artifact["default_values"]

st.info(
    f"Selected Model: {model_name} | "
    f"Decision Threshold: {threshold:.4f}"
)

st.divider()

# ==========================================
# 4. PREPARE TEST DATASET
# ==========================================

X = df[feature_names].copy()

y = df["diagnosis"].copy()

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
    st.error("Invalid or missing diagnosis labels in dataset.")
    st.stop()

y = y.astype(int)

# Reproduce original train/test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. SELECT TESTING METHOD
# ==========================================

# ==========================================
# PREDICTION METHOD SELECTION
# ==========================================

st.divider()

st.subheader("🔍 Choose Prediction Method")

method_col1, method_col2 = st.columns(2)

with method_col1:
    st.markdown("**✍️ Manual Prediction**")
    st.caption("Enter tumour measurements manually.")

with method_col2:
    st.markdown("**📊 Dataset Testing**")
    st.caption("Select an existing dataset record.")

testing_mode = st.radio(
    "Choose your prediction method:",
    [
        "Manual Input",
        "Test Using Dataset Samples"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

selected_record = None
actual_diagnosis = None

if testing_mode == "Test Using Dataset Samples":

    test_indices = X_test.index.tolist()

    selected_record = st.selectbox(
        "Select a test dataset record:",
        options=test_indices,
        format_func=lambda idx: (
            f"Test Record {idx + 1} - "
            f"Actual: "
            f"{'Malignant' if y.loc[idx] == 1 else 'Benign'}"
        )
    )

    sample_values = X_test.loc[selected_record]

    actual_diagnosis = int(y.loc[selected_record])

    st.success(
        f"Actual dataset diagnosis: "
        f"{'Malignant' if actual_diagnosis == 1 else 'Benign'}"
    )

else:

    sample_values = pd.Series(default_values)

# ==========================================
# 6. FEATURE GROUPING
# ==========================================

feature_groups = {
    "Mean Measurements": {
        "features": feature_names[:10],
        "description": "Average values of the tumour measurements."
    },

    "Standard Error Measurements": {
        "features": feature_names[10:20],
        "description": "Variability and standard error of the measurements."
    },

    "Worst Measurements": {
        "features": feature_names[20:30],
        "description": "Largest or worst values of the tumour measurements."
    }
}

# ==========================================
# 7. INPUT FORM
# ==========================================

st.divider()

st.subheader("🧬 Tumour Measurements")

st.caption(
    "Enter the 30 numerical measurements below to generate "
    "an experimental Benign or Malignant classification."
)

user_input = {}

with st.form("prediction_form"):

    for group_name, group_data in feature_groups.items():

        features = group_data["features"]
        description = group_data["description"]

        with st.expander(
            group_name,
            expanded=(group_name == "Mean Measurements")
        ):

            st.caption(description)

            columns = st.columns(2)

            for i, feature in enumerate(features):

                with columns[i % 2]:

                    value = st.number_input(
                        label=feature.replace("_", " ").title(),

                        value=float(sample_values[feature]),

                        min_value=0.0,

                        step=0.01,

                        format="%.5f",

                        disabled=(
                            testing_mode ==
                            "Test Using Dataset Samples"
                        ),

                        key=(
                            f"{selected_model}_{feature}_"
                            f"{testing_mode}_"
                            f"{selected_record if selected_record is not None else 'manual'}"
                        )
                    )

                    user_input[feature] = value

    submitted = st.form_submit_button(
        "🔍 Predict Diagnosis",
        use_container_width=True
    )



# ==========================================
# 8. PROFESSIONAL PREDICTION RESULTS
# ==========================================

if submitted:

    input_df = pd.DataFrame(
        [user_input],
        columns=feature_names
    )

    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    malignant_index = classes.index(1)
    benign_index = classes.index(0)

    malignant_probability = float(
        probabilities[malignant_index]
    )

    benign_probability = float(
        probabilities[benign_index]
    )

    # Apply the selected model's threshold
    prediction = int(
        malignant_probability >= threshold
    )

    predicted_label = (
        "Malignant" if prediction == 1 else "Benign"
    )

    predicted_probability = float(
        probabilities[classes.index(prediction)]
    )

    st.divider()

    st.subheader("📊 Prediction Results")

    st.caption(
        f"Prediction generated using {model_name}"
    )

    # Main prediction result
    if prediction == 1:

        st.error(
            "⚠️ Experimental Prediction: MALIGNANT"
        )

    else:

        st.success(
            "✓ Experimental Prediction: BENIGN"
        )

    # Probability metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Malignant Probability",
            f"{malignant_probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Predicted-Class Probability",
            f"{predicted_probability * 100:.2f}%"
        )

    # Probability visualization
    st.write("**Estimated Class Probabilities**")

    probability_data = pd.DataFrame({
        "Class": ["Benign", "Malignant"],
        "Estimated Probability": [
            benign_probability,
            malignant_probability
        ]
    })

    st.bar_chart(
        probability_data.set_index("Class")
    )

    st.caption(
        "Probabilities are model estimates and are not "
        "a measure of clinical certainty."
    )

    # ======================================
    # 9. ACTUAL VS PREDICTED COMPARISON
    # ======================================

    if testing_mode == "Test Using Dataset Samples":

        st.divider()

        st.subheader("Actual vs Predicted Comparison")

        actual_label = (
            "Malignant" if actual_diagnosis == 1
            else "Benign"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Actual Dataset Diagnosis",
                actual_label
            )

        with col2:
            st.metric(
                "Model Prediction",
                predicted_label
            )

        if prediction == actual_diagnosis:

            st.success(
                "Correct Prediction! The model prediction "
                "matches the dataset label."
            )

        else:

            st.error(
                "Incorrect Prediction. The model prediction "
                "does not match the dataset label."
            )

    # ======================================
    # 10. MEDICAL DISCLAIMER
    # ======================================

    st.divider()

    st.warning(
        "Medical Disclaimer: This application is an "
        "experimental machine learning project for "
        "educational purposes only. It is not clinically "
        "validated and must not be used for medical "
        "diagnosis or treatment decisions. Consult a "
        "qualified healthcare professional."
    )


    # ======================================
    # 10. MEDICAL DISCLAIMER
    # ======================================

    st.warning(
        "Medical Disclaimer: This application is an "
        "experimental machine learning project for "
        "educational purposes only. It is not clinically "
        "validated and must not be used for medical "
        "diagnosis or treatment decisions. Consult a "
        "qualified healthcare professional."
    )
