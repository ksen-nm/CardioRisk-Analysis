import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Page Config
st.set_page_config(page_title="CardioRisk Analytics", layout="wide", page_icon="🫀")

# Title
st.title("🫀 CardioRisk Analytics Dashboard")
st.markdown("An end-to-end cardiovascular risk analytics pipeline. *This is for educational and analytics purposes only and should not be used for medical diagnosis.*")

# Cache data loading
@st.cache_data
def load_data():
    df = None
    if os.path.exists("../tableau/cardio_dashboard_data.csv"):
        df = pd.read_csv("../tableau/cardio_dashboard_data.csv")
    elif os.path.exists("tableau/cardio_dashboard_data.csv"):
        df = pd.read_csv("tableau/cardio_dashboard_data.csv")
    return df

@st.cache_data
def load_segments():
    df = None
    if os.path.exists("../tableau/risk_segment_summary.csv"):
        df = pd.read_csv("../tableau/risk_segment_summary.csv")
    elif os.path.exists("tableau/risk_segment_summary.csv"):
        df = pd.read_csv("tableau/risk_segment_summary.csv")
    return df

@st.cache_data
def load_model_results():
    df = None
    if os.path.exists("../tableau/model_results.csv"):
        df = pd.read_csv("../tableau/model_results.csv")
    elif os.path.exists("tableau/model_results.csv"):
        df = pd.read_csv("tableau/model_results.csv")
    return df

@st.cache_resource
def load_best_model():
    model = None
    if os.path.exists("../models/best_model.pkl"):
        model = joblib.load("../models/best_model.pkl")
    elif os.path.exists("models/best_model.pkl"):
        model = joblib.load("models/best_model.pkl")
    return model

df = load_data()
df_segments = load_segments()
df_models = load_model_results()
model = load_best_model()

if df is None:
    st.warning("Data not found. Please ensure you have run the pipeline scripts first.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["A. Overview", "B. Risk Factor Analysis", "C. Model Performance", "D. Threshold Simulator", "E. Risk Segments", "F. Individual Prediction Demo"])

if page == "A. Overview":
    st.header("Executive Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("CVD Prevalence", f"{(df['cardio'].mean()*100):.1f}%")
    col3.metric("Average Age", f"{df['age_years'].mean():.1f}")
    col4.metric("Average BMI", f"{df['BMI'].mean():.1f}")
    col5.metric("Avg Systolic BP", f"{df['ap_hi'].mean():.0f}")
    
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x='cardio', palette='Set2', ax=ax)
    ax.set_xticklabels(['No CVD (0)', 'CVD Present (1)'])
    st.pyplot(fig)

elif page == "B. Risk Factor Analysis":
    st.header("Risk Factor Analysis")
    
    # Filters
    st.sidebar.subheader("Filters")
    age_filter = st.sidebar.multiselect("Age Group", df['age_group'].unique(), default=df['age_group'].unique())
    gender_filter = st.sidebar.multiselect("Gender", df['gender'].unique(), default=df['gender'].unique())
    
    df_filtered = df[(df['age_group'].isin(age_filter)) & (df['gender'].isin(gender_filter))]
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("CVD Rate by Age Group")
        rate_age = df_filtered.groupby('age_group')['cardio'].mean().reset_index()
        fig1, ax1 = plt.subplots()
        sns.barplot(data=rate_age, x='age_group', y='cardio', palette='Blues_d', ax=ax1)
        st.pyplot(fig1)
        
        st.subheader("CVD Rate by Cholesterol")
        rate_chol = df_filtered.groupby('cholesterol')['cardio'].mean().reset_index()
        fig3, ax3 = plt.subplots()
        sns.barplot(data=rate_chol, x='cholesterol', y='cardio', palette='Oranges_d', ax=ax3)
        st.pyplot(fig3)
        
    with col2:
        st.subheader("CVD Rate by BMI Category")
        rate_bmi = df_filtered.groupby('bmi_category')['cardio'].mean().reset_index()
        fig2, ax2 = plt.subplots()
        sns.barplot(data=rate_bmi, x='bmi_category', y='cardio', palette='Greens_d', ax=ax2)
        st.pyplot(fig2)
        
        st.subheader("Blood Pressure Distribution")
        fig4, ax4 = plt.subplots()
        sns.histplot(data=df_filtered, x='ap_hi', hue='cardio', kde=True, bins=30, ax=ax4)
        ax4.set_xlim(80, 200)
        st.pyplot(fig4)

elif page == "C. Model Performance":
    st.header("Model Performance")
    
    if df_models is not None:
        best_row = df_models.loc[df_models['F1-score'].idxmax()]
        
        st.subheader(f"Best Model: {best_row['Model']}")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{best_row['Accuracy']:.3f}")
        col2.metric("Precision", f"{best_row['Precision']:.3f}")
        col3.metric("Recall/Sensitivity", f"{best_row['Recall/Sensitivity']:.3f}")
        col4.metric("F1-score", f"{best_row['F1-score']:.3f}")
        col5.metric("ROC-AUC", f"{best_row['ROC-AUC']:.3f}")
        
        st.subheader("Model Comparison")
        st.dataframe(df_models.style.highlight_max(axis=0, color='lightgreen'))
    else:
        st.info("Model results file not found.")

elif page == "D. Threshold Simulator":
    st.header("Decision Threshold Simulator")
    st.markdown("Adjust the threshold to see the tradeoff between Sensitivity (Recall) and Specificity/Precision.")
    
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    
    y_true = df['cardio']
    y_prob = df['predicted_probability']
    y_pred = (y_prob >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{prec:.3f}")
    col2.metric("Recall (Sensitivity)", f"{rec:.3f}")
    col3.metric("Specificity", f"{spec:.3f}")
    col4.metric("F1-Score", f"{f1:.3f}")
    
    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap([[tn, fp], [fn, tp]], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])
        st.pyplot(fig)
        
    with col_cm2:
        st.subheader("Impact Explanation")
        if threshold < 0.5:
            st.write("📉 **Lowering the threshold:** You catch more potential high-risk cases (Higher Recall), but you also increase the number of healthy people flagged as at-risk (False Positives).")
        elif threshold > 0.5:
            st.write("📈 **Raising the threshold:** You reduce false alarms (Higher Precision/Specificity), but you miss more people who are actually at risk (False Negatives).")

elif page == "E. Risk Segments":
    st.header("Patient Risk Segments")
    
    if df_segments is not None:
        col1, col2, col3 = st.columns(3)
        low_count = df_segments.loc[df_segments['Risk Segment'] == 'Low Risk', 'Number of Patients'].values[0]
        med_count = df_segments.loc[df_segments['Risk Segment'] == 'Medium Risk', 'Number of Patients'].values[0]
        high_count = df_segments.loc[df_segments['Risk Segment'] == 'High Risk', 'Number of Patients'].values[0]
        
        col1.metric("Low Risk Patients", f"{low_count:,}")
        col2.metric("Medium Risk Patients", f"{med_count:,}")
        col3.metric("High Risk Patients", f"{high_count:,}")
        
        st.subheader("Segment Comparison")
        st.dataframe(df_segments)
        
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(data=df_segments, x='Risk Segment', y='Actual CVD Rate', order=['Low Risk', 'Medium Risk', 'High Risk'], ax=ax)
        ax.set_title("Actual CVD Rate by Segment")
        st.pyplot(fig)
    else:
        st.info("Risk segment summary not found.")

elif page == "F. Individual Prediction Demo":
    st.header("Individual Prediction Demo")
    st.warning("⚠️ **Disclaimer:** This prediction is for educational analytics demonstration only and should not be used for medical diagnosis.")
    
    if model is None:
        st.error("Model not loaded. Ensure train_model.py has been run.")
        st.stop()
        
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (Years)", 20, 100, 50)
        height = st.number_input("Height (cm)", 120, 220, 165)
        weight = st.number_input("Weight (kg)", 40, 200, 70)
        gender = st.selectbox("Gender", [1, 2], format_func=lambda x: "Women" if x==1 else "Men")
        ap_hi = st.number_input("Systolic Blood Pressure (ap_hi)", 70, 250, 120)
        ap_lo = st.number_input("Diastolic Blood Pressure (ap_lo)", 40, 150, 80)
        
    with col2:
        cholesterol = st.selectbox("Cholesterol", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        gluc = st.selectbox("Glucose", [1, 2, 3], format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
        smoke = st.selectbox("Smoker", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        alco = st.selectbox("Alcohol Intake", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        active = st.selectbox("Physically Active", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
    if st.button("Predict Risk"):
        # Engineer features
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        high_cholesterol = 1 if cholesterol > 1 else 0
        high_glucose = 1 if gluc > 1 else 0
        
        input_data = pd.DataFrame([{
            'gender': gender,
            'height': height,
            'weight': weight,
            'ap_hi': ap_hi,
            'ap_lo': ap_lo,
            'cholesterol': cholesterol,
            'gluc': gluc,
            'smoke': smoke,
            'alco': alco,
            'active': active,
            'age_years': age,
            'BMI': bmi,
            'pulse_pressure': pulse_pressure,
            'high_cholesterol': high_cholesterol,
            'high_glucose': high_glucose
        }])
        
        prob = model.predict_proba(input_data)[0, 1]
        
        if prob < 0.30:
            segment = "Low Risk"
            color = "green"
        elif prob <= 0.60:
            segment = "Medium Risk"
            color = "orange"
        else:
            segment = "High Risk"
            color = "red"
            
        st.markdown(f"### Predicted Risk Probability: <span style='color:{color}'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
        st.markdown(f"### Risk Segment: <span style='color:{color}'>{segment}</span>", unsafe_allow_html=True)
        
        st.write(f"The model estimates a {prob*100:.1f}% probability of cardiovascular disease presence based on the input factors. Patients in this range are typically classified as {segment}.")
