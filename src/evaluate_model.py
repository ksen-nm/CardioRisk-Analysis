import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.train_model import prepare_data

def evaluate_thresholds(model, X_test, y_test, thresholds=[0.30, 0.40, 0.50, 0.60, 0.70]):
    """
    Evaluate the impact of different decision thresholds on model performance.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results.append({
            'Threshold': t,
            'Precision': prec,
            'Recall/Sensitivity': rec,
            'Specificity': spec,
            'F1-score': f1,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'True Negatives': tn
        })
        
    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        _, X_test, _, y_test = prepare_data()
        
        df_thresholds = evaluate_thresholds(model, X_test, y_test)
        os.makedirs("reports", exist_ok=True)
        df_thresholds.to_csv("reports/threshold_analysis.csv", index=False)
        print("Threshold analysis saved to reports/threshold_analysis.csv")
        print(df_thresholds)
    else:
        print("Model not found. Run train_model.py first.")
