import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from src.train_model import prepare_data

def get_feature_importance(model, X_train):
    """
    Extract feature importance if the model supports it.
    """
    if hasattr(model, 'best_estimator_'):
        pipeline = model.best_estimator_
    else:
        pipeline = model
        
    classifier = pipeline.named_steps['model']
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        features = X_train.columns
        df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
        df_imp = df_imp.sort_values(by='Importance', ascending=False)
        return df_imp
    return None

def get_permutation_importance(model, X_test, y_test):
    """
    Compute permutation importance.
    """
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    df_imp = pd.DataFrame({'Feature': X_test.columns, 'Importance': result.importances_mean})
    df_imp = df_imp.sort_values(by='Importance', ascending=False)
    return df_imp

if __name__ == "__main__":
    model_path = "models/best_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        X_train, X_test, y_train, y_test = prepare_data()
        
        # We just test the functions
        feat_imp = get_feature_importance(model, X_train)
        if feat_imp is not None:
            print("Feature Importance:")
            print(feat_imp)
            
        perm_imp = get_permutation_importance(model, X_test, y_test)
        print("\nPermutation Importance:")
        print(perm_imp)
    else:
        print("Model not found. Run train_model.py first.")
