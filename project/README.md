# Healthcare Test Results Prediction

## Overview

A complete end-to-end Data Analytics and Machine Learning project built on a synthetic Healthcare Dataset.
The project predicts the **Test Result** of a hospital patient (Normal, Abnormal, or Inconclusive) based on
clinical, demographic, and administrative features.

---

## Problem Statement

Hospitals generate large volumes of patient data with every admission. The goal of this project is to
determine whether machine learning can automatically classify a patient's test result based on readily
available information such as age, medical condition, admission type, medication, and insurance provider.

---

## Objective

- Perform a thorough Exploratory Data Analysis (EDA) of the healthcare dataset
- Build and compare multiple multiclass classification models
- Identify the best-performing model and save it for deployment
- Provide an interactive Streamlit dashboard for real-time predictions

---

## Dataset

| Property        | Value                               |
|-----------------|-------------------------------------|
| Dataset Name    | Healthcare Dataset                  |
| Source          | Kaggle (Synthetic Healthcare Data)  |
| Total Records   | 55,500                              |
| After Dedup     | 54,966                              |
| Features        | 15 original → 13 engineered         |
| Target Variable | Test Results                        |
| Target Classes  | Normal, Abnormal, Inconclusive      |
| Problem Type    | Multiclass Classification           |

---

## Technologies Used

| Category       | Technology              |
|----------------|-------------------------|
| Language       | Python 3.9+             |
| Data Analysis  | pandas, numpy           |
| Visualization  | matplotlib, seaborn     |
| ML             | scikit-learn            |
| Model Saving   | joblib                  |
| Frontend       | Streamlit               |
| Reporting      | python-docx             |
| Notebook       | Jupyter                 |

---

## Project Structure

```
project/
│
├── data/
│   └── healthcare_dataset.csv
│
├── notebooks/
│   └── Data_Analytics_ML_Project.ipynb
│
├── model/
│   ├── trained_model.pkl
│   └── model_metadata.json
│
├── frontend/
│   └── app.py
│
├── outputs/
│   ├── figures/            (15 charts)
│   ├── metrics/            (model comparison, classification report, feature importance)
│   └── predictions/        (test_predictions.csv)
│
├── report/
│   └── Project_Report.docx
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

## Methodology

```
Dataset → Data Understanding → Data Cleaning → EDA → Feature Engineering
→ Preprocessing → Train/Test Split → Model Training → Evaluation
→ Model Comparison → Best Model Selection → Model Saving → Frontend
```

---

## Data Preprocessing

- Removed 534 duplicate rows (54,966 clean records remain)
- Extracted date features: Length of Stay, Admission Year, Month, Day-of-Week
- Dropped high-cardinality identifier columns: Name, Doctor, Hospital
- Numerical features: StandardScaler via Pipeline
- Categorical features: OrdinalEncoder via ColumnTransformer
- No missing values in the dataset
- 108 negative billing amounts retained (statistically valid outliers)

---

## EDA

Key findings:
- Target classes are well-balanced (~33% each)
- Age ranges from 13–89, mean ~51.5 years
- 6 medical conditions, each with ~9,200 patients
- 3 admission types equally distributed
- Billing amounts range from ~$0 to ~$52,700 (mean ~$25,500)
- Length of stay: 1–30 days (mean ~15.5 days)
- Admissions span May 2019 – May 2024

---

## Machine Learning Models

Four models were trained and evaluated:

| Model                    | Accuracy | F1-Score | ROC-AUC |
|--------------------------|----------|----------|---------|
| Random Forest            | 0.4263   | 0.4261   | 0.6313  |
| HistGradientBoosting     | 0.3413   | 0.3389   | 0.5163  |
| Gradient Boosting        | 0.3333   | 0.3325   | 0.5018  |
| Logistic Regression      | 0.3326   | 0.3304   | 0.4979  |

---

## Model Evaluation

Best Model: **Random Forest Classifier**

```
Classification Report:
              precision    recall  f1-score   support
    Abnormal       0.42      0.46      0.44      3688
Inconclusive       0.42      0.41      0.41      3640
      Normal       0.44      0.41      0.43      3666
    accuracy                           0.43     10994
```

**Note on accuracy**: The dataset is balanced across three classes. Random baseline = ~33.3%.
Random Forest achieves ~42.6% accuracy, indicating it has learned some signal from the features,
but the inherent difficulty is high because the dataset is synthetic and the features (demographics,
medication, insurance) have limited predictive power for test results.

---

## Final Model

**Selected Model**: Random Forest Classifier  
**Reason**: Highest accuracy (42.6%), highest F1-Score (42.6%), and highest ROC-AUC (0.63)
among all four trained models. Also provides feature importance for interpretability.

---

## How to Install

```bash
pip install -r requirements.txt
```

---

## How to Run the Pipeline

```bash
cd project
python run_pipeline.py
```

---

## How to Run Notebook

```bash
cd project
jupyter notebook notebooks/Data_Analytics_ML_Project.ipynb
```

---

## How to Run Frontend

```bash
cd project
streamlit run frontend/app.py
```

Open your browser at: http://localhost:8501

---

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 42.63% |
| Precision | 42.66% |
| Recall    | 42.63% |
| F1-Score  | 42.61% |
| ROC-AUC   | 0.6313 |

---

## Future Scope

- Collect real (non-synthetic) clinical lab result data for better signal
- Include lab test values (blood count, glucose, cholesterol) as features
- Apply SMOTE or other techniques if class imbalance arises
- Experiment with XGBoost, LightGBM, or neural network classifiers
- Hyperparameter tuning with GridSearchCV or Optuna
- Deploy the Streamlit app to Streamlit Cloud or Docker

---

*Project developed as part of an AI/Data Analytics internship project.*
