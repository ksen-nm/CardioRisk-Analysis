import pandas as pd
import numpy as np
import os

def engineer_features(df):
    """
    Create new features for cardiovascular risk prediction.
    - BMI and BMI categories
    - Age groups
    - Blood pressure categories
    - Pulse pressure
    - High cholesterol / glucose indicators
    """
    df = df.copy()
    
    # BMI
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # BMI category
    conditions = [
        (df['BMI'] < 18.5),
        (df['BMI'] >= 18.5) & (df['BMI'] < 25),
        (df['BMI'] >= 25) & (df['BMI'] < 30),
        (df['BMI'] >= 30)
    ]
    choices = ['underweight', 'normal', 'overweight', 'obese']
    df['bmi_category'] = np.select(conditions, choices, default='unknown')
    
    # Age groups
    age_conditions = [
        (df['age_years'] < 40),
        (df['age_years'] >= 40) & (df['age_years'] < 50),
        (df['age_years'] >= 50) & (df['age_years'] < 60),
        (df['age_years'] >= 60)
    ]
    age_choices = ['under_40', '40_49', '50_59', '60_plus']
    df['age_group'] = np.select(age_conditions, age_choices, default='unknown')
    
    # Blood pressure category (AHA guidelines approximation)
    bp_conditions = [
        (df['ap_hi'] < 120) & (df['ap_lo'] < 80),
        (df['ap_hi'] >= 120) & (df['ap_hi'] < 130) & (df['ap_lo'] < 80),
        ((df['ap_hi'] >= 130) & (df['ap_hi'] < 140)) | ((df['ap_lo'] >= 80) & (df['ap_lo'] < 90)),
        (df['ap_hi'] >= 140) | (df['ap_lo'] >= 90)
    ]
    bp_choices = ['normal', 'elevated', 'stage_1', 'stage_2']
    # Add crisis if very high
    crisis_mask = (df['ap_hi'] >= 180) | (df['ap_lo'] >= 120)
    
    df['bp_category'] = np.select(bp_conditions, bp_choices, default='unknown')
    df.loc[crisis_mask, 'bp_category'] = 'crisis'
    
    # Pulse pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # High cholesterol indicator (1 is normal, 2 and 3 are above normal)
    df['high_cholesterol'] = (df['cholesterol'] > 1).astype(int)
    
    # High glucose indicator
    df['high_glucose'] = (df['gluc'] > 1).astype(int)
    
    return df

if __name__ == "__main__":
    clean_path = "data/processed/cardio_cleaned.csv"
    if os.path.exists(clean_path):
        df_clean = pd.read_csv(clean_path)
        df_features = engineer_features(df_clean)
        df_features.to_csv("data/processed/cardio_features.csv", index=False)
        print("Feature engineering complete. File saved to data/processed/cardio_features.csv")
        print(f"Dataset shape: {df_features.shape}")
    else:
        print(f"Cleaned data not found at {clean_path}. Run data_cleaning.py first.")
