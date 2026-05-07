#!/bin/bash
set -e

cd /Users/kayn/Desktop/study/projects/Cardiovascular\ Risk\ Analytics\ Platform/cardiorisk-analytics

export PYTHONPATH=$(pwd)

echo "1. Data Cleaning..."
python src/data_cleaning.py

echo "2. Feature Engineering..."
python src/feature_engineering.py

echo "3. Data Validation..."
python src/data_validation.py

echo "4. Model Training (This might take a while)..."
python src/train_model.py

echo "5. Model Evaluation..."
python src/evaluate_model.py

echo "6. Explainability..."
python src/explain_model.py

echo "7. Predictions and Segmentation..."
python src/predict.py

echo "Pipeline complete."
