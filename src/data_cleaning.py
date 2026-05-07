import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load raw dataset."""
    return pd.read_csv(filepath, sep=';' if 'train' in filepath else ',') # usually cardiovascular dataset is semicolon separated

def clean_data(df):
    """
    Clean the cardiovascular disease dataset.
    - Drops 'id' column
    - Converts age from days to years
    - Removes duplicates
    - Filters invalid blood pressure values
    - Filters unrealistic height and weight values
    """
    initial_rows = len(df)
    report = []
    report.append({"Step": "Initial Data", "Rows Remaining": initial_rows, "Rows Removed": 0})
    
    # Drop id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Convert age
    if 'age' in df.columns:
        df['age_years'] = (df['age'] / 365.25).astype(int)
        df = df.drop(columns=['age'])
        
    # Remove duplicates
    df = df.drop_duplicates()
    dupes_removed = initial_rows - len(df)
    report.append({"Step": "Remove Duplicates", "Rows Remaining": len(df), "Rows Removed": dupes_removed})
    
    # Filter blood pressure
    prev_rows = len(df)
    df = df[
        (df['ap_hi'] > 0) & 
        (df['ap_lo'] > 0) & 
        (df['ap_hi'] > df['ap_lo']) & 
        (df['ap_hi'] <= 250) & 
        (df['ap_lo'] <= 200) &
        (df['ap_hi'] >= 70) & # Reasonable lower bound
        (df['ap_lo'] >= 40)
    ]
    bp_removed = prev_rows - len(df)
    report.append({"Step": "Filter Blood Pressure", "Rows Remaining": len(df), "Rows Removed": bp_removed})
    
    # Filter Height and Weight using reasonable domains (e.g. height 120-220cm, weight 40-200kg)
    prev_rows = len(df)
    df = df[
        (df['height'] >= 120) & (df['height'] <= 220) &
        (df['weight'] >= 40) & (df['weight'] <= 200)
    ]
    hw_removed = prev_rows - len(df)
    report.append({"Step": "Filter Height & Weight", "Rows Remaining": len(df), "Rows Removed": hw_removed})
    
    report_df = pd.DataFrame(report)
    return df, report_df

if __name__ == "__main__":
    # Test execution
    raw_path = "data/raw/cardio_train.csv"
    if os.path.exists(raw_path):
        df_raw = load_data(raw_path)
        # Handle delimiter issue: Kaggle cardio_train.csv is usually ';' separated
        if df_raw.shape[1] == 1:
            df_raw = pd.read_csv(raw_path, sep=';')
            
        df_clean, quality_report = clean_data(df_raw)
        
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        df_clean.to_csv("data/processed/cardio_cleaned.csv", index=False)
        quality_report.to_csv("reports/data_quality_report.csv", index=False)
        print("Data cleaning complete.")
        print(quality_report)
    else:
        print(f"Raw data not found at {raw_path}")
