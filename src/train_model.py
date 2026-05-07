import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def prepare_data(filepath="data/processed/cardio_features.csv"):
    df = pd.read_csv(filepath)
    # Use numeric features + encoded categorical features
    # Drop string categorical columns created for EDA
    drop_cols = ['bmi_category', 'age_group', 'bp_category', 'cardio']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['cardio']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_models():
    # Defining a smaller grid for faster execution if needed, particularly for SVM
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), {'model__C': [0.1, 1.0, 10.0]}),
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [5, 10, None]}),
        'Random Forest': (RandomForestClassifier(random_state=42), {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}),
        'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {'model__n_estimators': [50, 100], 'model__learning_rate': [0.05, 0.1]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [5, 10]}),
        'SVM': (SVC(probability=True, random_state=42, max_iter=2000), {'model__C': [1.0]}), # Limit max_iter to prevent hanging
        'LDA': (LinearDiscriminantAnalysis(), {}),
        'QDA': (QuadraticDiscriminantAnalysis(), {})
    }
    return models

def train_and_evaluate():
    X_train, X_test, y_train, y_test = prepare_data()
    models = build_models()
    
    results = []
    best_model = None
    best_f1 = 0
    best_name = ""
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    for name, (model, params) in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Use GridSearchCV for most, but keep it simple to ensure it runs
        search = GridSearchCV(pipeline, param_grid=params, cv=3, scoring='f1', n_jobs=-1)
        search.fit(X_train, y_train)
        
        # Evaluate
        y_pred = search.predict(X_test)
        if hasattr(search, "predict_proba"):
            y_prob = search.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_prob)
        else:
            roc_auc = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall/Sensitivity': rec,
            'Specificity': specificity,
            'F1-score': f1,
            'ROC-AUC': roc_auc
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = search
            best_name = name
            
    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/model_results.csv", index=False)
    
    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"Best model: {best_name} saved to models/best_model.pkl")
    print(results_df)
    
    return results_df, best_model, X_test, y_test

if __name__ == "__main__":
    train_and_evaluate()
