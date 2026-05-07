import pandas as pd
import numpy as np
import os

def validate_data(df):
    """
    Validate the processed dataset against business and logical constraints.
    Returns a dictionary of check results.
    """
    results = {}
    
    # No duplicate rows
    results['no_duplicates'] = not df.duplicated().any()
    
    # No negative values in specific columns
    non_negative_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'BMI', 'pulse_pressure']
    results['no_negatives'] = (df[non_negative_cols] >= 0).all().all()
    
    # ap_hi > ap_lo
    results['valid_bp_range'] = (df['ap_hi'] > df['ap_lo']).all()
    
    # cardio must only contain 0 or 1
    if 'cardio' in df.columns:
        results['valid_cardio'] = set(df['cardio'].unique()).issubset({0, 1})
        
    # smoke, alco, active only 0 or 1
    binary_cols = ['smoke', 'alco', 'active']
    results['valid_binary_flags'] = all(set(df[col].unique()).issubset({0, 1}) for col in binary_cols if col in df.columns)
    
    # cholesterol, gluc only 1, 2, 3
    cat_cols = ['cholesterol', 'gluc']
    results['valid_cat_levels'] = all(set(df[col].unique()).issubset({1, 2, 3}) for col in cat_cols if col in df.columns)
    
    # BMI in reasonable range (e.g., 10 to 55)
    results['reasonable_bmi'] = (df['BMI'] >= 10).all() and (df['BMI'] <= 55).all()
    
    # No missing values
    results['no_missing_values'] = not df.isnull().any().any()
    
    return results

if __name__ == "__main__":
    features_path = "data/processed/cardio_features.csv"
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        val_results = validate_data(df)
        
        # Save to data quality report
        report_path = "reports/data_quality_report.csv"
        
        # We append to existing if possible, or create a new block
        val_df = pd.DataFrame(list(val_results.items()), columns=['Validation Check', 'Passed'])
        print("Data Validation Results:")
        print(val_df)
        
        if os.path.exists(report_path):
            existing_report = pd.read_csv(report_path)
            # Just append the validation info to the end or overwrite with a new file
            val_df.to_csv("reports/validation_summary.csv", index=False)
            print("Validation summary saved to reports/validation_summary.csv")
        else:
            val_df.to_csv(report_path, index=False)
    else:
        print(f"Features data not found at {features_path}.")
