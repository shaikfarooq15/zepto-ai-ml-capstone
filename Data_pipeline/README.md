# Analytics

This module explores the Titanic dataset, performs EDA, trains classification and regression models, and saves the ML pipelines.

## Setup

    pip install -r requirements.txt
    python module1_data_pipeline/titanic_analysis.py
    python module1_data_pipeline/model_pipeline.py

## Main Work

- Data cleaning and missing-value handling
- Exploratory Data Analysis
- Feature standardization
- Classification
- Class imbalance handling
- Random Forest tuning
- Fare regression
- Pipeline saving and reloading

## EDA Results

Cleaned dataset:

    889 rows

Outliers:

    Age  = 65
    Fare = 114

Fare statistics:

    Mean   = 32.097
    Median = 14.454
    Mode   = 8.050

Survival rates:

- Female: 0.740
- Male: 0.189
- First Class: 0.626
- Second Class: 0.473
- Third Class: 0.242

Strongest correlations:

    Fare ↔ Pclass = -0.548
    Parch ↔ SibSp =  0.415

## Model Results

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|------------|----------|-----------|--------|-----|-----|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | 0.861 |
| Decision Tree | 0.770 | 0.690 | 0.721 | 0.705 | 0.754 |
| Random Forest | 0.803 | 0.771 | 0.691 | 0.729 | 0.823 |
| Tuned Random Forest | 0.809 | 0.783 | 0.691 | 0.734 | 0.825 |

## Imbalance Handling

Balanced Logistic Regression:

    Accuracy  = 79.21%
    Precision = 71.83%
    Recall    = 75.00%
    F1 Score  = 73.38%

SMOTE Logistic Regression:

    Accuracy  = 79.78%
    Precision = 73.53%
    Recall    = 73.53%
    F1 Score  = 73.53%

## Random Forest Tuning

Best parameters:

    max_depth = 10
    max_features = sqrt
    n_estimators = 50

OOB Score:

    81.01%

## Fare Regression

Model:

    Linear Regression

Results:

    MAE         = 21.14
    RMSE        = 41.75
    R²          = 0.3468
    Adjusted R² = 0.3118

## Saved Pipelines

Classification and regression pipelines were saved using Joblib and successfully reloaded for prediction.

## Project Structure

    module1_data_pipeline/
    ├── titanic_analysis.py
    ├── model_pipeline.py
    ├── titanic.csv
    └── README.md

## Technologies

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib

## Status

Analytics is complete.