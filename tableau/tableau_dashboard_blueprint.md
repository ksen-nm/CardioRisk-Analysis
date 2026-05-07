# Tableau Dashboard Blueprint

This document outlines the step-by-step instructions for building a professional Tableau dashboard using the CSV exports from the CardioRisk Analytics project.

## Required Datasets
Import the following files from the `tableau/` folder into Tableau:
1. `cardio_dashboard_data.csv` (Main dataset for patient-level visualizations)
2. `risk_segment_summary.csv` (Aggregated metrics by risk segment)
3. `model_results.csv` (Model performance comparison)
4. `threshold_analysis.csv` (Threshold tuning metrics)

---

## Calculated Fields to Create in Tableau

Before building the dashboards, create the following calculated fields in Tableau using the `cardio_dashboard_data.csv` source:

**CVD Rate:**
```tableau
AVG([cardio])
```

**BMI:**
```tableau
[weight] / (([height] / 100) ^ 2)
```

**Risk Segment:**
```tableau
IF [predicted_probability] < 0.30 THEN "Low Risk"
ELSEIF [predicted_probability] <= 0.60 THEN "Medium Risk"
ELSE "High Risk"
END
```

**Age Group:**
```tableau
IF [age_years] < 40 THEN "Under 40"
ELSEIF [age_years] < 50 THEN "40-49"
ELSEIF [age_years] < 60 THEN "50-59"
ELSE "60+"
END
```

**BMI Category:**
```tableau
IF [BMI] < 18.5 THEN "Underweight"
ELSEIF [BMI] < 25 THEN "Normal"
ELSEIF [BMI] < 30 THEN "Overweight"
ELSE "Obese"
END
```

**Blood Pressure Category:**
```tableau
IF [ap_hi] < 120 AND [ap_lo] < 80 THEN "Normal"
ELSEIF [ap_hi] < 130 AND [ap_lo] < 80 THEN "Elevated"
ELSEIF [ap_hi] < 140 OR [ap_lo] < 90 THEN "Stage 1"
ELSEIF [ap_hi] < 180 OR [ap_lo] < 120 THEN "Stage 2"
ELSE "Crisis Range"
END
```

**False Negative Flag:**
```tableau
IF [cardio] = 1 AND [predicted_label] = 0 THEN 1 ELSE 0 END
```

**False Positive Flag:**
```tableau
IF [cardio] = 0 AND [predicted_label] = 1 THEN 1 ELSE 0 END
```

---

## Dashboard Pages

### Page 1: Executive Overview
**Purpose:** High-level summary of the patient population and CVD prevalence.

- **KPI cards:**
  - Total patients (Count of `id` or rows)
  - CVD prevalence (`CVD Rate` formatted as percentage)
  - Average age (Average of `age_years`)
  - Average BMI (Average of `BMI`)
  - Average systolic BP (Average of `ap_hi`)
- **Charts:**
  - **Target distribution:** Pie chart or donut chart showing the count of patients by `cardio` (0 vs 1).
  - **CVD rate by age group:** Bar chart with `Age Group` on Columns and `CVD Rate` on Rows.
  - **CVD rate by cholesterol level:** Bar chart with `cholesterol` on Columns and `CVD Rate` on Rows.

### Page 2: Risk Factor Analysis
**Purpose:** Deep dive into how specific risk factors correlate with cardiovascular disease.

- **Charts:**
  - **CVD rate by BMI category:** Bar chart of `CVD Rate` by `BMI Category`.
  - **CVD rate by blood pressure category:** Bar chart of `CVD Rate` by `Blood Pressure Category`.
  - **CVD rate by glucose level:** Bar chart of `CVD Rate` by `gluc`.
  - **CVD rate by physical activity:** Bar chart of `CVD Rate` by `active`.
  - **Average systolic BP by CVD status:** Box plot or bar chart comparing `ap_hi` for `cardio` = 0 vs `cardio` = 1.
- **Filters (Apply to all charts on page):**
  - `gender`
  - `Age Group`
  - `cholesterol`
  - `gluc`
  - `BMI Category`
  - `Risk Segment`

### Page 3: Model Performance
**Purpose:** Compare the different machine learning models and evaluate the best one.
*Use the `model_results.csv` data source for this page.*

- **KPI cards:**
  - Best model (Filter for the model with the best recall or F1)
  - Accuracy
  - Precision
  - Recall/Sensitivity
  - F1-score
  - ROC-AUC
- **Charts:**
  - **Model comparison bar chart:** `Model` on Columns, `Recall` and `Accuracy` on Rows (dual axis).
  - **Accuracy vs recall comparison:** Scatter plot of `Accuracy` vs `Recall` with `Model` on detail.
  - **Confusion matrix table:** Use the best model's predictions from `cardio_dashboard_data.csv` to build a text table of Actual (`cardio`) vs Predicted (`predicted_label`).

### Page 4: Threshold Analysis
**Purpose:** Illustrate the business tradeoff between catching high-risk patients and avoiding false positives.
*Use the `threshold_analysis.csv` data source for this page.*

- **Charts:**
  - **Threshold vs precision:** Line chart of `Threshold` vs `Precision`.
  - **Threshold vs recall:** Line chart of `Threshold` vs `Recall`.
  - **Threshold vs false positives:** Line chart of `Threshold` vs `False Positives`.
  - **Threshold vs false negatives:** Line chart of `Threshold` vs `False Negatives`.
- **Explanation Text Box:**
  - "Lower thresholds catch more potential high-risk cases but increase false positives."
  - "Higher thresholds reduce false positives but may miss more high-risk cases."

### Page 5: Risk Segmentation
**Purpose:** Summarize the characteristics of the population grouped by predicted risk level.
*Use the `risk_segment_summary.csv` or calculated fields on `cardio_dashboard_data.csv`.*

- **KPI cards:**
  - Low-risk count
  - Medium-risk count
  - High-risk count
- **Charts:**
  - **Count of patients by risk segment:** Bar chart.
  - **Actual CVD rate by risk segment:** Bar chart (`Risk Segment` vs `CVD Rate`).
  - **Average age by risk segment:** Bar chart.
  - **Average BMI by risk segment:** Bar chart.
  - **Average systolic BP by risk segment:** Bar chart.
  - **Cholesterol distribution by risk segment:** 100% stacked bar chart showing the percentage of each cholesterol level within each risk segment.
- **Table:**
  - Full segment summary text table showing all metrics side-by-side for Low, Medium, and High Risk.
