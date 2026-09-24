# Data Pipeline

This module builds a complete machine learning pipeline using the Titanic dataset.

The pipeline covers:

- Data Preparation
- Exploratory Data Analysis
- Classification
- Imbalance Handling
- Hyperparameter Tuning
- Regression
- Model Evaluation
- Pipeline Saving and Reloading

---

## 1. Project Objective

The goal is to analyze the Titanic dataset and build machine learning models for:

1. Survival classification
2. Fare prediction

The workflow includes preprocessing, model training, evaluation, tuning, and pipeline persistence.

---

## 2. Data Preparation

Dataset:

    Titanic Dataset

The dataset was cleaned by:

- Handling missing values
- Removing unnecessary columns
- Handling outliers
- Preparing numerical and categorical features
- Standardizing numerical features

The cleaned dataset was saved as:

    titanic.csv

---

## 3. Exploratory Data Analysis

EDA included:

- Univariate analysis
- Bivariate analysis
- Multivariate analysis
- Correlation analysis
- Survival analysis
- Outlier analysis
- Standardization

---

## 4. Classification

The target variable:

    survived

Models used:

- Logistic Regression
- Decision Tree
- Random Forest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## 5. Imbalanced Data

Different approaches were compared:

- Baseline Logistic Regression
- Class-weight Balanced Logistic Regression
- SMOTE Logistic Regression

---

## 6. Hyperparameter Tuning

Random Forest was tuned using:

    GridSearchCV

Parameters:

- `n_estimators`
- `max_depth`
- `max_features`

The tuning used:

- 5-fold cross-validation
- F1 Score
- Out-of-Bag (OOB) evaluation

---

## 7. Regression

The regression target:

    fare

Evaluation metrics:

- MAE
- RMSE
- R²
- Adjusted R²
- Residual Analysis

The regression pipeline was saved and reloaded for prediction.

---

## 8. Project Structure

    module2_machine_learning/
    ├── titanic_analysis.py
    ├── model_pipeline.py
    ├── titanic.csv
    └── README.md

---

## 9. Technologies

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib

---

## 10. Pipeline

    Titanic Dataset
        ↓
    Data Preparation
        ↓
    Exploratory Data Analysis
        ↓
    Classification
        ↓
    Imbalance Handling
        ↓
    Hyperparameter Tuning
        ↓
    Regression
        ↓
    Model Evaluation
        ↓
    Save / Reload Pipelines

---

