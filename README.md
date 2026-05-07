# CardioRisk Analytics: Cardiovascular Risk Prediction, Risk Factor Analysis, Model Explainability, and Dashboard Reporting

## Project Overview
Built an end-to-end cardiovascular risk analytics pipeline on 70,000 patient records, combining data cleaning, statistical analysis, machine learning model comparison, threshold tuning, explainability, and interactive dashboarding to support risk-factor analysis and decision-ready healthcare insights.

## Business/Analytics Problem
Cardiovascular diseases (CVDs) are the leading cause of death globally. Early detection and proactive management of risk factors are critical to improving patient outcomes. This project aims to identify the key patient attributes associated with CVD and develop a predictive model that categorizes patients into risk segments. The insights generated are intended to help healthcare analysts and decision-makers understand population risk profiles, not for direct clinical diagnosis.

## Dataset Description
The dataset used is the [Cardiovascular Disease dataset from Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data), containing 70,000 patient records.
It includes 11 input features and a binary target variable `cardio` (1 = cardiovascular disease present, 0 = absent).

**Features:**
- `age`: Age (originally in days, converted to years)
- `height`: Height in cm
- `weight`: Weight in kg
- `gender`: Categorical code (1 = women, 2 = men)
- `ap_hi`: Systolic blood pressure
- `ap_lo`: Diastolic blood pressure
- `cholesterol`: 1 = normal, 2 = above normal, 3 = well above normal
- `gluc`: 1 = normal, 2 = above normal, 3 = well above normal
- `smoke`: Binary smoking indicator
- `alco`: Binary alcohol intake indicator
- `active`: Binary physical activity indicator

## Key Questions Answered
- Which patient attributes are most associated with cardiovascular disease?
- How do different machine learning models compare for risk prediction?
- What tradeoff exists between accuracy, sensitivity, precision, and false negatives?
- Which patient groups fall into low, medium, and high-risk segments?
- How can model outputs be translated into actionable healthcare analytics insights?

## Project Workflow
1. **Data Cleaning & Validation:** Handling invalid blood pressure values, unrealistic height/weight measurements, and deduplication.
2. **Feature Engineering:** Creating BMI categories, age groups, blood pressure categories, and calculating pulse pressure.
3. **Exploratory Data Analysis (EDA) & Statistical Analysis:** Visualizing risk factor distributions and running chi-square/t-tests to identify significant predictors.
4. **Machine Learning Modeling:** Comparing multiple classifiers (Logistic Regression, Random Forest, Gradient Boosting, etc.) with hyperparameter tuning and cross-validation.
5. **Threshold Tuning & Explainability:** Analyzing the tradeoff between false positives and false negatives, and using feature importance/SHAP to explain risk drivers.
6. **Risk Segmentation:** Grouping patients into Low, Medium, and High-risk segments based on predicted probabilities.
7. **Dashboarding & Reporting:** Building a Streamlit application and preparing Tableau-ready exports for executive summaries.

## Repository Structure
```
cardiorisk-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── MODEL_CARD.md
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability_and_segmentation.ipynb
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_validation.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── explain_model.py
│   └── predict.py
├── reports/
├── dashboard/
│   └── app.py
├── tableau/
└── models/
```

## Methods Used
- Data Wrangling & Feature Engineering
- Statistical Hypothesis Testing (Chi-Square, t-tests)
- Machine Learning (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, KNN, SVM, LDA, QDA)
- Cross-Validation & Grid/Randomized Search
- Classification Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Threshold Tuning
- Model Explainability (Feature Importance, Permutation Importance, SHAP)
- Patient Risk Segmentation
- Interactive Dashboarding (Streamlit)

## Model Results
We evaluated several models emphasizing **recall/sensitivity** to minimize false negatives (missing a high-risk patient). 
The best models achieve around ~73-74% accuracy. We deliberately tuned the decision threshold to favor higher sensitivity, balancing the tradeoff between catching potential risk cases and minimizing false positives. Detailed results and threshold curves are available in the modeling notebooks and Streamlit dashboard.

## Key Insights
- **Top Risk Drivers:** Systolic blood pressure (`ap_hi`), age, cholesterol, and BMI emerged as the strongest predictors of cardiovascular disease.
- **Risk Segmentation:** High-risk patients consistently show elevated average systolic blood pressure, higher BMI, an older age distribution, and elevated cholesterol levels compared to the low-risk segment.

## Streamlit Dashboard

![Dashboard Demo](assets/demo_v2.webp)

An interactive Streamlit application (`dashboard/app.py`) provides:
- **Overview & Risk Factor Analysis:** Explore CVD prevalence by demographics.
- **Model Performance & Threshold Simulator:** Visualize confusion matrices dynamically based on custom decision thresholds.
- **Risk Segments:** Review aggregated profiles of low, medium, and high-risk patient groups.
- **Individual Prediction Demo:** Input patient data to see predicted risk probabilities and risk segment classification.

## Tableau Reporting Exports
The project generates Tableau-ready CSV exports (`tableau/` directory) and a detailed blueprint (`tableau/tableau_dashboard_blueprint.md`) for creating executive-level BI dashboards encompassing KPIs, risk factor analysis, model performance, and patient segmentation.

## How to Run the Project
1. **Clone the repository.**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Add Raw Data:** Place `cardio_train.csv` in `data/raw/`.
4. **Run Pipeline Scripts:** (Alternatively, run the notebooks in sequence).
5. **Launch Streamlit Dashboard:** `streamlit run dashboard/app.py`

## Limitations
- **Educational Use Only:** This model is not clinically validated. It should not be used for diagnosis, treatment decisions, or replacing medical professionals. It is intended solely as a data science and analytics demonstration.
- **Data Quality:** The raw dataset contains some extreme outliers and potentially synthetic or self-reported anomalies that required heuristic cleaning.

## Future Improvements
- Incorporate more complex explainability techniques (like local SHAP for all test instances).
- Integrate additional clinical features if available (e.g., family history, ECG results).
- Deploy the Streamlit app to a cloud platform.

## Portfolio Highlights
- Built an end-to-end healthcare analytics pipeline on 70,000 patient records.
- Performed data cleaning, validation, BMI feature engineering, and statistical risk-factor analysis.
- Compared multiple machine learning models using cross-validation and healthcare-relevant metrics.
- Applied threshold tuning to evaluate false positive and false negative tradeoffs.
- Used feature importance and explainability methods to identify major cardiovascular risk drivers.
- Created risk segmentation outputs for low, medium, and high-risk patient groups.
- Built a Streamlit dashboard for risk exploration, model performance review, and individual prediction demo.
- Prepared Tableau-ready exports and a dashboard blueprint for executive-style reporting.
