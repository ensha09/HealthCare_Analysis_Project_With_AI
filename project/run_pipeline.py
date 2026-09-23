"""
Healthcare Dataset - Complete ML Pipeline
Multiclass Classification: Predict Test Results (Normal / Abnormal / Inconclusive)
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================
import os, warnings, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

FIGURES_DIR   = 'outputs/figures'
METRICS_DIR   = 'outputs/metrics'
PREDS_DIR     = 'outputs/predictions'
MODEL_DIR     = 'model'

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PREDS_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR,   exist_ok=True)

print("=" * 60)
print("HEALTHCARE TEST RESULTS PREDICTION")
print("Problem Type: Multiclass Classification")
print("=" * 60)

# ============================================================
# SECTION 3 — LOAD DATASET
# ============================================================
df = pd.read_csv('data/healthcare_dataset.csv')
print(f"\nDataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nStatistical summary:")
print(df.describe(include='all'))

# ============================================================
# SECTION 4 — DATA QUALITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("DATA QUALITY ANALYSIS")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count}")

print(f"\nNegative Billing Amounts: {(df['Billing Amount'] < 0).sum()}")

# Drop duplicates
df = df.drop_duplicates()
print(f"After dropping duplicates: {df.shape[0]} rows")

# ============================================================
# SECTION 5 — DATE FEATURE ENGINEERING & EDA
# ============================================================
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days
df['Admission Year']    = df['Date of Admission'].dt.year
df['Admission Month']   = df['Date of Admission'].dt.month
df['Admission DayOfWeek'] = df['Date of Admission'].dt.dayofweek

# Fix name casing (remove from model — just for data quality)
df['Name'] = df['Name'].str.title()

print("\n=== EDA ===")
print("\nTarget distribution (Test Results):")
print(df['Test Results'].value_counts())

# --- PLOT 1: Target class distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
vals = df['Test Results'].value_counts()
colors = ['#3b82d4', '#e05c5c', '#5cb85c']
ax.bar(vals.index, vals.values, color=colors, edgecolor='black')
ax.set_title('Distribution of Test Results (Target Variable)', fontsize=13)
ax.set_xlabel('Test Result')
ax.set_ylabel('Count')
for i, v in enumerate(vals.values):
    ax.text(i, v + 100, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/01_target_distribution.png', dpi=120)
plt.close()
print("Saved: 01_target_distribution.png")

# --- PLOT 2: Age distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Age'], bins=30, color='#3b82d4', edgecolor='black')
ax.set_title('Age Distribution of Patients', fontsize=13)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/02_age_distribution.png', dpi=120)
plt.close()
print("Saved: 02_age_distribution.png")

# --- PLOT 3: Billing Amount distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Billing Amount'], bins=40, color='#5cb85c', edgecolor='black')
ax.set_title('Billing Amount Distribution', fontsize=13)
ax.set_xlabel('Billing Amount ($)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/03_billing_distribution.png', dpi=120)
plt.close()
print("Saved: 03_billing_distribution.png")

# --- PLOT 4: Medical Condition counts ---
fig, ax = plt.subplots(figsize=(8, 4))
cond_counts = df['Medical Condition'].value_counts()
ax.barh(cond_counts.index, cond_counts.values, color='#7c5cd8', edgecolor='black')
ax.set_title('Patient Count by Medical Condition', fontsize=13)
ax.set_xlabel('Count')
ax.set_ylabel('Medical Condition')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/04_medical_condition.png', dpi=120)
plt.close()
print("Saved: 04_medical_condition.png")

# --- PLOT 5: Test Results by Medical Condition ---
fig, ax = plt.subplots(figsize=(10, 5))
ct = pd.crosstab(df['Medical Condition'], df['Test Results'])
ct.plot(kind='bar', ax=ax, color=['#e05c5c', '#f0ad4e', '#5cb85c'])
ax.set_title('Test Results by Medical Condition', fontsize=13)
ax.set_xlabel('Medical Condition')
ax.set_ylabel('Count')
ax.legend(title='Test Result')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/05_results_by_condition.png', dpi=120)
plt.close()
print("Saved: 05_results_by_condition.png")

# --- PLOT 6: Length of Stay distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df['Length of Stay'], bins=30, color='#f0ad4e', edgecolor='black')
ax.set_title('Length of Stay Distribution', fontsize=13)
ax.set_xlabel('Days')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/06_length_of_stay.png', dpi=120)
plt.close()
print("Saved: 06_length_of_stay.png")

# --- PLOT 7: Boxplot Age by Test Results ---
fig, ax = plt.subplots(figsize=(8, 5))
unique_results = list(df['Test Results'].unique())
groups = [df[df['Test Results'] == r]['Age'].values for r in unique_results]
bp = ax.boxplot(groups, patch_artist=True,
                boxprops=dict(facecolor='#3b82d4', color='black'))
ax.set_xticks(range(1, len(unique_results) + 1))
ax.set_xticklabels(unique_results)
ax.set_title('Age Distribution by Test Result', fontsize=13)
ax.set_xlabel('Test Result')
ax.set_ylabel('Age')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/07_age_by_test_result.png', dpi=120)
plt.close()
print("Saved: 07_age_by_test_result.png")

# --- PLOT 8: Admission Type distribution ---
fig, ax = plt.subplots(figsize=(7, 4))
at_counts = df['Admission Type'].value_counts()
ax.bar(at_counts.index, at_counts.values, color=['#3b82d4', '#e05c5c', '#5cb85c'], edgecolor='black')
ax.set_title('Admission Type Distribution', fontsize=13)
ax.set_xlabel('Admission Type')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/08_admission_type.png', dpi=120)
plt.close()
print("Saved: 08_admission_type.png")

# --- PLOT 9: Correlation heatmap (numerical) ---
num_cols = ['Age', 'Billing Amount', 'Room Number', 'Length of Stay',
            'Admission Year', 'Admission Month', 'Admission DayOfWeek']
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, linewidths=0.5)
ax.set_title('Correlation Heatmap (Numerical Features)', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/09_correlation_heatmap.png', dpi=120)
plt.close()
print("Saved: 09_correlation_heatmap.png")

# --- PLOT 10: Gender distribution ---
fig, ax = plt.subplots(figsize=(5, 4))
gender_counts = df['Gender'].value_counts()
ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
       colors=['#3b82d4', '#e05c5c'], startangle=140)
ax.set_title('Gender Distribution', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/10_gender_distribution.png', dpi=120)
plt.close()
print("Saved: 10_gender_distribution.png")

# --- PLOT 11: Billing Amount by Test Result ---
fig, ax = plt.subplots(figsize=(8, 5))
for tr in df['Test Results'].unique():
    ax.hist(df[df['Test Results'] == tr]['Billing Amount'], bins=30, alpha=0.6, label=tr)
ax.set_title('Billing Amount by Test Result', fontsize=13)
ax.set_xlabel('Billing Amount ($)')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/11_billing_by_result.png', dpi=120)
plt.close()
print("Saved: 11_billing_by_result.png")

# --- PLOT 12: Admissions over time ---
monthly = df.groupby(df['Date of Admission'].dt.to_period('M')).size()
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(monthly.index.astype(str), monthly.values, color='#3b82d4', linewidth=1.5)
ax.set_title('Monthly Hospital Admissions Over Time', fontsize=13)
ax.set_xlabel('Month')
ax.set_ylabel('Admissions')
plt.xticks(rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/12_admissions_over_time.png', dpi=120)
plt.close()
print("Saved: 12_admissions_over_time.png")

# ============================================================
# SECTION 6 — DATA PREPROCESSING & FEATURE SELECTION
# ============================================================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Drop columns that are identifiers / high-cardinality with no predictive value
DROP_COLS = ['Name', 'Doctor', 'Hospital', 'Date of Admission', 'Discharge Date']

TARGET = 'Test Results'

feature_df = df.drop(columns=DROP_COLS)
X = feature_df.drop(columns=[TARGET])
y = feature_df[TARGET]

print(f"\nFeatures used ({len(X.columns)}): {X.columns.tolist()}")
print(f"Target: {TARGET}")
print(f"\nClass distribution:\n{y.value_counts()}")

# Save feature names for frontend
feature_names = X.columns.tolist()

# Define column types
numerical_features = ['Age', 'Billing Amount', 'Room Number', 'Length of Stay',
                      'Admission Year', 'Admission Month', 'Admission DayOfWeek']
categorical_features = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider',
                        'Admission Type', 'Medication']

# Pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# ============================================================
# SECTION 8 — TRAIN/TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ============================================================
# SECTION 9 — TRAIN MULTIPLE MODELS
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
}

results = []
trained_pipelines = {}

for name, clf in models.items():
    print(f"\nTraining: {name} ...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ROC-AUC (one-vs-rest, probability)
    try:
        y_prob = pipe.predict_proba(X_test)
        from sklearn.preprocessing import LabelBinarizer
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)
        roc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
    except Exception:
        roc = np.nan

    results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(roc, 4) if not np.isnan(roc) else np.nan
    })
    trained_pipelines[name] = pipe
    print(f"  Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={roc:.4f}")

# ============================================================
# SECTION 10 — MODEL EVALUATION & COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(f'{METRICS_DIR}/model_comparison.csv', index=False)
print(f"\nSaved: {METRICS_DIR}/model_comparison.csv")

# Comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
width = 0.18
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors_bar = ['#3b82d4', '#5cb85c', '#f0ad4e', '#e05c5c']
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_bar)):
    ax.bar(x + i * width, results_df[metric], width, label=metric, color=color, edgecolor='black')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
ax.set_ylim(0, 1.0)
ax.set_title('Model Comparison — Classification Metrics', fontsize=13)
ax.set_ylabel('Score')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/13_model_comparison.png', dpi=120)
plt.close()
print("Saved: 13_model_comparison.png")

# ============================================================
# SECTION 11 — BEST MODEL SELECTION
# ============================================================
best_row = results_df.iloc[0]
best_name = best_row['Model']
best_pipe = trained_pipelines[best_name]
print(f"\nBest model: {best_name}")
print(f"  Accuracy : {best_row['Accuracy']}")
print(f"  F1-Score : {best_row['F1-Score']}")
print(f"  ROC-AUC  : {best_row['ROC-AUC']}")

# Detailed classification report
y_pred_best = best_pipe.predict(X_test)
report_str = classification_report(y_test, y_pred_best)
print(f"\nClassification Report ({best_name}):\n{report_str}")
with open(f'{METRICS_DIR}/classification_report.txt', 'w') as f:
    f.write(f"Best Model: {best_name}\n\n")
    f.write(report_str)
print(f"Saved: {METRICS_DIR}/classification_report.txt")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best, labels=best_pipe.classes_)
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_pipe.classes_)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_name}', fontsize=13)
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/14_confusion_matrix.png', dpi=120)
plt.close()
print("Saved: 14_confusion_matrix.png")

# Feature importance (if available)
try:
    clf_step = best_pipe.named_steps['classifier']
    if hasattr(clf_step, 'feature_importances_'):
        all_feat_names = numerical_features + categorical_features
        importances = clf_step.feature_importances_
        feat_imp_df = pd.DataFrame({'Feature': all_feat_names, 'Importance': importances})
        feat_imp_df = feat_imp_df.sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='#3b82d4', edgecolor='black')
        ax.set_title(f'Feature Importance — {best_name}', fontsize=13)
        ax.set_xlabel('Importance')
        plt.tight_layout()
        plt.savefig(f'{FIGURES_DIR}/15_feature_importance.png', dpi=120)
        plt.close()
        print("Saved: 15_feature_importance.png")
        feat_imp_df.to_csv(f'{METRICS_DIR}/feature_importance.csv', index=False)
except Exception as e:
    print(f"Feature importance not available: {e}")

# ============================================================
# SECTION 12 — SAVE MODEL
# ============================================================
model_path = f'{MODEL_DIR}/trained_model.pkl'
joblib.dump(best_pipe, model_path)
print(f"\nModel saved: {model_path}")

# Save metadata
metadata = {
    'model_name': best_name,
    'target': TARGET,
    'classes': list(best_pipe.classes_),
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'feature_names': feature_names,
    'metrics': {
        'accuracy': float(best_row['Accuracy']),
        'f1_score': float(best_row['F1-Score']),
        'roc_auc': float(best_row['ROC-AUC']) if not pd.isna(best_row['ROC-AUC']) else None
    },
    'all_models': results
}
with open(f'{MODEL_DIR}/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Metadata saved: {MODEL_DIR}/model_metadata.json")

# ============================================================
# SECTION 13 — TEST SAVED MODEL
# ============================================================
print("\n" + "=" * 60)
print("TESTING SAVED MODEL")
print("=" * 60)
loaded_model = joblib.load(model_path)
sample = X_test.iloc[:5].copy()
sample_pred = loaded_model.predict(sample)
sample_actual = y_test.iloc[:5].values
print("Sample predictions vs actual:")
for pred, actual in zip(sample_pred, sample_actual):
    match = "OK" if pred == actual else "WRONG"
    print(f"  Predicted: {pred:15s} | Actual: {actual:15s} [{match}]")

# Save predictions
y_pred_all = loaded_model.predict(X_test)
pred_df = X_test.copy()
pred_df['Actual'] = y_test.values
pred_df['Predicted'] = y_pred_all
pred_df.to_csv(f'{PREDS_DIR}/test_predictions.csv', index=False)
print(f"\nPredictions saved: {PREDS_DIR}/test_predictions.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Dataset        : Healthcare Dataset")
print(f"Rows (cleaned) : {df.shape[0]}")
print(f"Target         : {TARGET} (3 classes)")
print(f"Problem Type   : Multiclass Classification")
print(f"Best Model     : {best_name}")
print(f"Accuracy       : {best_row['Accuracy']}")
print(f"F1-Score       : {best_row['F1-Score']}")
print(f"ROC-AUC        : {best_row['ROC-AUC']}")
print("=" * 60)
