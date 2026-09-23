"""
Healthcare Test Results Prediction - Streamlit Frontend
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# ── Paths ────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'trained_model.pkl')
META_PATH   = os.path.join(BASE_DIR, 'model', 'model_metadata.json')
FIGURES_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')
DATA_PATH   = os.path.join(BASE_DIR, 'data', 'healthcare_dataset.csv')
METRICS_PATH = os.path.join(BASE_DIR, 'outputs', 'metrics', 'model_comparison.csv')

st.set_page_config(
    page_title="Healthcare Test Results Predictor",
    page_icon="🏥",
    layout="wide"
)

# ── Load model & metadata ─────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    return model, meta

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates()
    df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
    df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days
    return df

try:
    model, meta = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error  = str(e)

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Hospital_Sign.svg/200px-Hospital_Sign.svg.png",
                 width=60, use_container_width=False)
st.sidebar.title("Healthcare ML App")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "Home & Dataset Info",
    "Predict Test Result",
    "EDA Visualizations",
    "Model Performance"
])

# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "Home & Dataset Info":
    st.title("🏥 Healthcare Test Results Prediction Dashboard")
    st.markdown("""
    **Project**: Predicting patient test results using clinical and demographic data.  
    **Problem Type**: Multiclass Classification (Normal / Abnormal / Inconclusive)  
    **Dataset**: Synthetic Healthcare Dataset from Kaggle
    """)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",   f"{len(df):,}")
    col2.metric("Features Used",   "13")
    col3.metric("Target Classes",  "3")
    col4.metric("Best Model",      meta['model_name'] if model_loaded else "N/A")

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Class Distribution (Target: Test Results)")
    vc = df['Test Results'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(vc.index, vc.values, color=['#3b82d4', '#e05c5c', '#5cb85c'], edgecolor='black')
    ax.set_xlabel('Test Result')
    ax.set_ylabel('Count')
    ax.set_title('Test Results Distribution')
    for i, v in enumerate(vc.values):
        ax.text(i, v + 100, str(v), ha='center')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════════════
elif page == "Predict Test Result":
    st.title("🔬 Predict Patient Test Result")
    st.markdown("Fill in the patient details below and click **Predict** to get the test result prediction.")

    if not model_loaded:
        st.error(f"Model could not be loaded: {model_error}")
        st.stop()

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age    = st.number_input("Age",   min_value=1, max_value=120, value=45)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        blood_type = st.selectbox("Blood Type", ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])

    with col2:
        st.subheader("Clinical Info")
        medical_condition = st.selectbox("Medical Condition", sorted(df['Medical Condition'].unique()))
        admission_type    = st.selectbox("Admission Type",    sorted(df['Admission Type'].unique()))
        medication        = st.selectbox("Medication",        sorted(df['Medication'].unique()))

    with col3:
        st.subheader("Administrative")
        insurance = st.selectbox("Insurance Provider", sorted(df['Insurance Provider'].unique()))
        billing   = st.number_input("Billing Amount ($)", min_value=0.0, max_value=100000.0, value=25000.0, step=100.0)
        room_num  = st.number_input("Room Number", min_value=100, max_value=500, value=200)
        los       = st.number_input("Length of Stay (days)", min_value=1, max_value=60, value=15)

    st.markdown("---")
    adm_year  = st.number_input("Admission Year",  min_value=2015, max_value=2030, value=2023)
    adm_month = st.slider("Admission Month", 1, 12, 6)
    adm_dow   = st.selectbox("Admission Day of Week", [0, 1, 2, 3, 4, 5, 6],
                              format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])

    if st.button("🔍 Predict Test Result", type="primary"):
        input_data = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Medical Condition': medical_condition,
            'Insurance Provider': insurance,
            'Billing Amount': billing,
            'Room Number': room_num,
            'Admission Type': admission_type,
            'Medication': medication,
            'Length of Stay': los,
            'Admission Year': adm_year,
            'Admission Month': adm_month,
            'Admission DayOfWeek': adm_dow
        }])

        try:
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            color_map = {'Normal': 'green', 'Abnormal': 'red', 'Inconclusive': 'orange'}
            pred_color = color_map.get(prediction, 'blue')

            st.markdown(f"""
            <div style="background-color:#f7f8fa; border: 2px solid {pred_color};
                        border-radius:10px; padding:20px; text-align:center;">
                <h2 style="color:{pred_color};">Predicted Test Result: {prediction}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Prediction Probabilities")
            prob_df = pd.DataFrame({'Class': classes, 'Probability': [f"{p:.2%}" for p in probabilities]})
            st.table(prob_df)

            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(classes, probabilities,
                    color=[color_map.get(c, '#3b82d4') for c in classes], edgecolor='black')
            ax2.set_ylabel('Probability')
            ax2.set_title('Class Probability Distribution')
            ax2.set_ylim(0, 1)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — EDA VISUALIZATIONS
# ══════════════════════════════════════════════════════════════
elif page == "EDA Visualizations":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical", "Financial", "Time Analysis"])

    with tab1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Age'], bins=30, color='#3b82d4', edgecolor='black')
        ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.set_title('Patient Age Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Gender Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        gc = df['Gender'].value_counts()
        ax.pie(gc.values, labels=gc.index, autopct='%1.1f%%', colors=['#3b82d4','#e05c5c'])
        ax.set_title('Gender Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        st.subheader("Medical Condition Distribution")
        mc = df['Medical Condition'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(mc.index, mc.values, color='#7c5cd8', edgecolor='black')
        ax.set_xlabel('Count'); ax.set_title('Patients by Medical Condition')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Test Results by Medical Condition")
        ct = pd.crosstab(df['Medical Condition'], df['Test Results'])
        fig, ax = plt.subplots(figsize=(10, 5))
        ct.plot(kind='bar', ax=ax, color=['#e05c5c','#f0ad4e','#5cb85c'])
        ax.set_title('Test Results by Medical Condition')
        ax.set_xlabel('Medical Condition'); ax.set_ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Length of Stay Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Length of Stay'], bins=30, color='#f0ad4e', edgecolor='black')
        ax.set_xlabel('Days'); ax.set_ylabel('Count'); ax.set_title('Length of Stay')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        st.subheader("Billing Amount Distribution")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Billing Amount'], bins=40, color='#5cb85c', edgecolor='black')
        ax.set_xlabel('Billing Amount ($)'); ax.set_ylabel('Count')
        ax.set_title('Billing Amount Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.subheader("Billing Amount by Test Result")
        fig, ax = plt.subplots(figsize=(8, 5))
        for tr in df['Test Results'].unique():
            ax.hist(df[df['Test Results'] == tr]['Billing Amount'],
                    bins=30, alpha=0.6, label=tr)
        ax.set_xlabel('Billing Amount ($)'); ax.set_ylabel('Count')
        ax.set_title('Billing by Test Result'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        st.subheader("Monthly Admissions Over Time")
        monthly = df.groupby(df['Date of Admission'].dt.to_period('M')).size()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(monthly.index.astype(str), monthly.values, color='#3b82d4', linewidth=1.5)
        ax.set_title('Monthly Hospital Admissions')
        ax.set_xlabel('Month'); ax.set_ylabel('Admissions')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("📈 Model Performance & Comparison")

    if model_loaded:
        st.subheader("Best Model")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model",    meta['model_name'])
        c2.metric("Accuracy", f"{meta['metrics']['accuracy']:.2%}")
        c3.metric("F1-Score", f"{meta['metrics']['f1_score']:.2%}")

    st.markdown("---")
    st.subheader("All Models Comparison")
    if os.path.exists(METRICS_PATH):
        comp_df = pd.read_csv(METRICS_PATH)
        st.dataframe(comp_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comp_df))
        width = 0.2
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors_bar = ['#3b82d4', '#5cb85c', '#f0ad4e', '#e05c5c']
        for i, (m, c) in enumerate(zip(metrics_to_plot, colors_bar)):
            ax.bar(x + i * width, comp_df[m], width, label=m, color=c, edgecolor='black')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comp_df['Model'], rotation=15, ha='right')
        ax.set_ylim(0, 0.7)
        ax.set_title('Model Comparison'); ax.set_ylabel('Score'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm_path = os.path.join(FIGURES_DIR, '14_confusion_matrix.png')
    if os.path.exists(cm_path):
        st.image(cm_path, caption=f"Confusion Matrix — {meta['model_name'] if model_loaded else 'Best Model'}")

    st.subheader("Feature Importance")
    fi_path = os.path.join(FIGURES_DIR, '15_feature_importance.png')
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Feature Importance")

    st.markdown("---")
    st.subheader("About the Model")
    st.info("""
    **Target Variable**: Test Results (Normal / Abnormal / Inconclusive)  
    **Problem Type**: Multiclass Classification  
    **Train/Test Split**: 80% / 20% (stratified)  
    **Preprocessing**: StandardScaler (numerical) + OrdinalEncoder (categorical)  
    **Features**: Age, Gender, Blood Type, Medical Condition, Insurance Provider,
    Billing Amount, Room Number, Admission Type, Medication, Length of Stay,
    Admission Year, Admission Month, Admission DayOfWeek  
    **Note**: The moderate accuracy reflects the inherent difficulty of predicting
    test outcomes from administrative/demographic data alone. The dataset is balanced
    across three classes (each ~33%), making random chance baseline ~33%.
    """)
