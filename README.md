# Breast Cancer Detection System Using Machine Learning

## 1. Project Overview

The Breast Cancer Detection System is a machine learning-based application developed to classify breast tumour records into two categories: Benign and Malignant.

The project uses the Breast Cancer Wisconsin Diagnostic (WDBC) dataset and applies machine learning algorithms to analyse numerical tumour measurements and predict the corresponding class.

A Streamlit web application provides an interactive interface where users can enter tumour measurements or test existing dataset records.

**This project is an educational prototype and is not intended for clinical diagnosis.**

## 2. Objectives

- Analyse the Breast Cancer Wisconsin Diagnostic dataset.
- Perform exploratory data analysis and data preprocessing.
- Train and compare multiple machine learning algorithms.
- Apply hyperparameter tuning to improve model performance.
- Evaluate models using classification metrics.
- Develop an interactive Streamlit prediction application.

## 3. Dataset

**Dataset:** Breast Cancer Wisconsin Diagnostic (WDBC)

- Total records: 569
- Input features: 30 numerical features
- Target variable: `diagnosis`
- Benign records: 357
- Malignant records: 212

The 30 features are grouped into:

1. Mean measurements
2. Standard error measurements
3. Worst measurements

Target encoding:

- 0 – Benign
- 1 – Malignant

Place the dataset at `data/breast_cancer_wdbc.csv`.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 5. Project Workflow

1. Dataset loading and inspection
2. Exploratory Data Analysis (EDA)
3. Missing-value and duplicate analysis
4. Feature distribution and correlation analysis
5. Outlier analysis
6. Data preprocessing
7. Train-test splitting
8. Machine learning model training
9. Hyperparameter tuning using GridSearchCV
10. Model evaluation and comparison
11. Final model integration
12. Streamlit application development

## 6. Machine Learning Algorithms

The project experiments with:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

Hyperparameter tuning is performed using GridSearchCV and stratified cross-validation.

## 7. Model Comparison

The following results were obtained for the three tuned models on the 114-record test set.

| Metric | Tuned Logistic Regression | Tuned Random Forest | Tuned SVM |
|---|---:|---:|---:|
| Accuracy | 97.37% | 97.37% | 97.37% |
| Precision | 97.56% | 100% | 100% |
| Malignant Recall | 95.24% | 92.86% | 92.86% |
| F1 Score | 96.39% | 96.30% | 96.30% |
| F2 Score | 95.69% | 94.20% | 94.20% |
| ROC-AUC | 99.54% | 99.50% | 99.27% |
| False Negatives | 2 | 3 | 3 |

Tuned Logistic Regression was selected as the recommended candidate because it achieved the highest malignant recall, F2 score, and ROC-AUC among these three tuned models.

The model comparison is experimental and does not establish clinical suitability.

## 8. Streamlit Application Features

- Selection between three trained machine learning models.
- Manual entry of 30 tumour measurements.
- Testing using existing dataset records.
- Benign and Malignant predictions.
- Malignant probability display.
- Predicted-class probability display.
- Actual-versus-predicted comparison for dataset samples.

## 9. Project Structure

```text
Breast_Cancer_Detection/
│
├── data/
│   ├── breast_cancer_wdbc.csv
│   └── processed/
│
├── models/
│   ├── final_breast_cancer_model.pkl
│   ├── best_breast_cancer_model.pkl
│   ├── baseline_svm_model.pkl
│   ├── tuned_logistic_regression.pkl
│   ├── tuned_random_forest.pkl
│   └── tuned_svm.pkl
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
├── reports/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/krishnamurthyu/Breast-Cancer-Detection-ML.git
cd Breast-Cancer-Detection-ML
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows PowerShell:

```powershell
venv\Scripts\Activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 11. Limitations

- The project uses a single diagnostic dataset.
- The model has not been clinically validated.
- Test-set performance does not guarantee performance on new populations.
- Further external validation and testing are required.

## 12. Future Scope

- Evaluate the system using independent external datasets.
- Explore additional machine learning and ensemble methods.
- Add explainability techniques such as SHAP or LIME.
- Improve the application interface.
- Investigate deployment and monitoring.

## 13. Disclaimer

This project is developed for educational and experimental purposes only. Its predictions must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 14. Author

**Author:** U Krushnamoorthy  
**Program:** B.Tech Computer Science and Engineering (AI & ML)  
**University:** Alliance University