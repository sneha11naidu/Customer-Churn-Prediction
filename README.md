# Customer-Churn-Prediction
Telecom Customer Churn Prediction: A Comparison of Naive Bayes and Decision Trees (MATLAB)

README.txt
Telecom Customer Churn Prediction – Naive Bayes vs Decision Tree (MATLAB)

Project Overview
This project implements and compares two supervised machine learning models — Naive Bayes and Decision Trees — for predicting customer churn using the Telco Customer Churn dataset. The code reproduces the final trained models, evaluates them on a held-out test set, and generates figures used in the accompanying poster.

Directory Structure
/
│── main.m
│── Telco-Customer-Churn.csv
│── Figures/
│── README.txt

Requirements
- MATLAB version: R2022a or later
- Required Toolboxes:
  - Statistics and Machine Learning Toolbox

How to Run the Code
1. Place all files in the same directory.
2. Open MATLAB and set this folder as the working directory.
3. Run: main
4. The script will preprocess the data, train both models, perform cross-validation and tuning, evaluate on the test set, and save figures.

Generated Figures
The script generates:
- Churn distribution
- Baseline confusion matrix
- Cross-validation accuracy per fold
- Decision Tree hyperparameter tuning heatmap
- Confusion matrices (NB & DT)
- ROC curves
- Precision–Recall curves
- Decision Tree feature importance

Dataset
Telco Customer Churn dataset (Kaggle):
https://www.kaggle.com/blastchar/telco-customer-churn

Notes
- Random seed fixed for reproducibility.
- Standardisation applied only where required.
- Strict train–validation–test separation maintained.
