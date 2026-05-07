import pandas as pd
import numpy as np
import joblib
import os
from src.train_model import prepare_data

def assign_risk_segments(y_prob):
    """
    Assign risk segments based on probability.
    - Low risk: < 0.30
    - Medium risk: 0.30 - 0.60
    - High risk: > 0.60
    """
    conditions = [
        (y_prob < 0.30),
        (y_prob >= 0.30) & (y_prob <= 0.60),
        (y_prob > 0.60)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    return np.select(conditions, choices, default='Unknown')

def generate_predictions():
    # Load model and data
    model = joblib.load("models/best_model.pkl")
    
    # We will score the entire dataset to build the Tableau export and segments
    df_full = pd.read_csv("data/processed/cardio_features.csv")
    
    drop_cols = ['bmi_category', 'age_group', 'bp_category', 'cardio']
    X_full = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns])
    
    y_prob = model.predict_proba(X_full)[:, 1]
    y_pred = model.predict(X_full)
    
    df_full['predicted_probability'] = y_prob
    df_full['predicted_label'] = y_pred
    df_full['Risk Segment'] = assign_risk_segments(y_prob)
    
    # Create segment summary
    segment_summary = []
    total_patients = len(df_full)
    
    for segment in ['Low Risk', 'Medium Risk', 'High Risk']:
        seg_data = df_full[df_full['Risk Segment'] == segment]
        if len(seg_data) == 0:
            continue
            
        segment_summary.append({
            'Risk Segment': segment,
            'Number of Patients': len(seg_data),
            'Percentage': len(seg_data) / total_patients * 100,
            'Actual CVD Rate': seg_data['cardio'].mean() if 'cardio' in seg_data.columns else np.nan,
            'Average Age': seg_data['age_years'].mean(),
            'Average BMI': seg_data['BMI'].mean(),
            'Average Systolic BP': seg_data['ap_hi'].mean(),
            'Average Diastolic BP': seg_data['ap_lo'].mean(),
            'Average Pulse Pressure': seg_data['pulse_pressure'].mean(),
            'Cholesterol 1 (Normal) %': (seg_data['cholesterol'] == 1).mean() * 100,
            'Cholesterol 2 (Above) %': (seg_data['cholesterol'] == 2).mean() * 100,
            'Cholesterol 3 (Well Above) %': (seg_data['cholesterol'] == 3).mean() * 100,
            'Glucose 1 (Normal) %': (seg_data['gluc'] == 1).mean() * 100,
            'Physical Activity Rate %': seg_data['active'].mean() * 100,
            'Smoking Rate %': seg_data['smoke'].mean() * 100,
            'Alcohol Intake Rate %': seg_data['alco'].mean() * 100
        })
        
    df_segment_summary = pd.DataFrame(segment_summary)
    
    os.makedirs("reports", exist_ok=True)
    os.makedirs("tableau", exist_ok=True)
    
    df_segment_summary.to_csv("reports/risk_segment_summary.csv", index=False)
    
    # Save tableau exports
    df_full.to_csv("tableau/cardio_dashboard_data.csv", index=False)
    df_segment_summary.to_csv("tableau/risk_segment_summary.csv", index=False)
    
    # Also copy model_results and threshold_analysis if they exist
    if os.path.exists("reports/model_results.csv"):
        pd.read_csv("reports/model_results.csv").to_csv("tableau/model_results.csv", index=False)
    if os.path.exists("reports/threshold_analysis.csv"):
        pd.read_csv("reports/threshold_analysis.csv").to_csv("tableau/threshold_analysis.csv", index=False)
        
    print("Predictions and segmentations complete. Tableau files created.")

if __name__ == "__main__":
    generate_predictions()
