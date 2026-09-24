# ==========================================
# MODULE 2 - MACHINE LEARNING
# Titanic Classification + Fare Regression
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.over_sampling import SMOTE


# ==========================================
# 1. LOAD TITANIC CSV
# ==========================================
# titanic_analysis.py downloads the dataset once and saves
# titanic.csv. This module uses that offline CSV.

df = pd.read_csv("titanic.csv")

print("\nDataset loaded from titanic.csv")
print("Dataset shape:", df.shape)

# ==========================================
# 2. CLEANING FOR MODELING
# ==========================================

df_clean = df.copy()

# Drop column with extremely high missingness.
df_clean = df_clean.drop(columns=["deck"])

# Drop the two rows where embarked information is missing.
df_clean = df_clean.dropna(
    subset=["embarked", "embark_town"]
)

print("\nCleaned dataset shape:", df_clean.shape)
print("Remaining missing values:")
print(df_clean.isnull().sum())


# ==========================================
# 3. CLASSIFICATION FEATURES
# ==========================================

features = [
    "age",
    "fare",
    "sex",
    "pclass",
    "sibsp",
    "parch",
    "embarked"
]

X = df_clean[features]
y = df_clean["survived"]

print("\nClassification features:")
print(features)

print("\nTarget distribution:")
print(y.value_counts())


# ==========================================
# 4. CLASSIFICATION TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nClassification train shape:", X_train.shape)
print("Classification test shape:", X_test.shape)


# ==========================================
# 5. LEAKAGE-SAFE PREPROCESSING
# ==========================================

numeric_features = [
    "age",
    "fare",
    "pclass",
    "sibsp",
    "parch"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Fit preprocessing ONLY on training data.
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the training rules to test data.
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)


# ==========================================
# 6. HELPER FUNCTION FOR CLASSIFICATION
# ==========================================

def evaluate_classifier(model, model_name):
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities)
    }

    print(f"\n========== {model_name.upper()} ==========")
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"AUC:       {metrics['AUC']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    fpr, tpr, _ = roc_curve(y_test, probabilities)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.show()

    return model, predictions, probabilities, metrics


# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================

logistic_model, logistic_pred, logistic_probability, logistic_metrics = (
    evaluate_classifier(
        LogisticRegression(max_iter=1000),
        "Logistic Regression"
    )
)


# ==========================================
# 8. DECISION TREE
# ==========================================

decision_tree, tree_pred, tree_probability, tree_metrics = (
    evaluate_classifier(
        DecisionTreeClassifier(random_state=42),
        "Decision Tree"
    )
)

print("\n========== DECISION TREE VISUALIZATION ==========")

feature_names = preprocessor.get_feature_names_out()

plt.figure(figsize=(20, 10))
plot_tree(
    decision_tree,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=3,
    fontsize=8
)
plt.title("Decision Tree - First Three Levels")
plt.show()


# ==========================================
# 9. RANDOM FOREST
# ==========================================

random_forest, forest_pred, forest_probability, forest_metrics = (
    evaluate_classifier(
        RandomForestClassifier(random_state=42),
        "Random Forest"
    )
)


# ==========================================
# 10. CLASS IMBALANCE
# ==========================================

print("\n========== CLASS IMBALANCE ==========")

print("\nTraining class distribution:")
print(y_train.value_counts())

# ------------------------------------------
# Balanced Logistic Regression
# ------------------------------------------

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train_processed, y_train)

balanced_pred = balanced_model.predict(X_test_processed)

balanced_metrics = {
    "Model": "Balanced Logistic Regression",
    "Accuracy": accuracy_score(y_test, balanced_pred),
    "Precision": precision_score(y_test, balanced_pred),
    "Recall": recall_score(y_test, balanced_pred),
    "F1 Score": f1_score(y_test, balanced_pred)
}

print("\nBalanced Logistic Regression:")
print(pd.Series(balanced_metrics))


# ------------------------------------------
# SMOTE
# ------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

smote_model = LogisticRegression(max_iter=1000)

smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test_processed)

smote_metrics = {
    "Model": "SMOTE Logistic Regression",
    "Accuracy": accuracy_score(y_test, smote_pred),
    "Precision": precision_score(y_test, smote_pred),
    "Recall": recall_score(y_test, smote_pred),
    "F1 Score": f1_score(y_test, smote_pred)
}

print("\nSMOTE Logistic Regression:")
print(pd.Series(smote_metrics))


# ==========================================
# 11. GRIDSEARCHCV - RANDOM FOREST
# ==========================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        oob_score=True
    ),
    param_grid,
    cv=5,
    scoring="f1"
)

grid_search.fit(X_train_processed, y_train)

best_model = grid_search.best_estimator_

print("\n========== GRID SEARCH ==========")
print("Best parameters:")
print(grid_search.best_params_)
print("Best cross-validation F1:", grid_search.best_score_)
print("Best model OOB score:", best_model.oob_score_)


# ==========================================
# 12. TUNED RANDOM FOREST
# ==========================================

best_pred = best_model.predict(X_test_processed)
best_probability = best_model.predict_proba(X_test_processed)[:, 1]

tuned_rf_metrics = {
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy_score(y_test, best_pred),
    "Precision": precision_score(y_test, best_pred),
    "Recall": recall_score(y_test, best_pred),
    "F1 Score": f1_score(y_test, best_pred),
    "AUC": roc_auc_score(y_test, best_probability)
}

print("\n========== TUNED RANDOM FOREST ==========")

for key, value in tuned_rf_metrics.items():
    if key != "Model":
        print(f"{key}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_pred))


# ==========================================
# 13. CLASSIFICATION MODEL COMPARISON
# ==========================================

classification_comparison = pd.DataFrame([
    logistic_metrics,
    tree_metrics,
    forest_metrics,
    tuned_rf_metrics
])

print("\n========== CLASSIFICATION MODEL COMPARISON ==========")
print(classification_comparison.round(4).to_string(index=False))


# ==========================================
# 14. FINAL CLASSIFICATION MODEL SELECTION
# ==========================================
# Select using F1 first and AUC as the tie-breaker.
# This is based on the actual test-set metrics produced above.

classification_comparison_sorted = classification_comparison.sort_values(
    by=["F1 Score", "AUC"],
    ascending=False
)

selected_model_name = classification_comparison_sorted.iloc[0]["Model"]

print("\n========== CLASSIFICATION MODEL SELECTION ==========")
print("Selection rule: highest test F1 Score, with AUC as tie-breaker.")
print("Selected model based on this rule:", selected_model_name)

print("""
The classification comparison reports Accuracy, Precision, Recall,
F1 Score and AUC for the tested models. The selection rule is stated
explicitly so the saved deployment model is based on calculated
evaluation metrics rather than hard-coded results.
""")


# ==========================================
# 15. FINAL CLASSIFICATION PIPELINE
# ==========================================

# Use the selected classifier object.
classifier_objects = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Tuned Random Forest": best_model
}

selected_classifier = classifier_objects[selected_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", selected_classifier)
])

final_pipeline.fit(X_train, y_train)

print("\nFinal classification pipeline trained successfully.")


# ==========================================
# 16. SAVE / RELOAD CLASSIFICATION PIPELINE
# ==========================================

joblib.dump(
    final_pipeline,
    "titanic_survival_pipeline.joblib"
)

print("Classification pipeline saved.")

loaded_pipeline = joblib.load(
    "titanic_survival_pipeline.joblib"
)

print("Classification pipeline reloaded successfully.")


# ==========================================
# 17. TEST CLASSIFICATION PIPELINE WITH RAW INPUT
# ==========================================

raw_passenger = pd.DataFrame([{
    "age": 22,
    "fare": 7.25,
    "sex": "male",
    "pclass": 3,
    "sibsp": 1,
    "parch": 0,
    "embarked": "S"
}])

prediction = loaded_pipeline.predict(raw_passenger)

print("\nRaw passenger:")
print(raw_passenger)

print("\nPredicted survival:")
print(prediction[0])


# ==========================================
# 18. FARE REGRESSION
# ==========================================

# Target = fare.
# Fare itself is NOT used as an input feature.

regression_features = [
    "age",
    "sex",
    "pclass",
    "sibsp",
    "parch",
    "embarked"
]

X_fare = df_clean[regression_features]
y_fare = df_clean["fare"]

print("\n========== FARE REGRESSION ==========")
print("Regression features:", regression_features)
print("X shape:", X_fare.shape)
print("y shape:", y_fare.shape)


# ==========================================
# 19. REGRESSION TRAIN / TEST SPLIT
# ==========================================

X_fare_train, X_fare_test, y_fare_train, y_fare_test = train_test_split(
    X_fare,
    y_fare,
    test_size=0.2,
    random_state=42
)

print("\nRegression train shape:", X_fare_train.shape)
print("Regression test shape:", X_fare_test.shape)


# ==========================================
# 20. REGRESSION PREPROCESSING
# ==========================================

reg_numeric_features = [
    "age",
    "pclass",
    "sibsp",
    "parch"
]

reg_categorical_features = [
    "sex",
    "embarked"
]

reg_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

reg_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

reg_preprocessor = ColumnTransformer([
    ("numeric", reg_numeric_pipeline, reg_numeric_features),
    ("categorical", reg_categorical_pipeline, reg_categorical_features)
])

X_fare_train_processed = reg_preprocessor.fit_transform(
    X_fare_train
)

X_fare_test_processed = reg_preprocessor.transform(
    X_fare_test
)

print("\nProcessed regression train shape:",
      X_fare_train_processed.shape)
print("Processed regression test shape:",
      X_fare_test_processed.shape)


# ==========================================
# 21. MULTIVARIATE LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_fare_train_processed,
    y_fare_train
)

y_fare_pred = linear_model.predict(
    X_fare_test_processed
)

print("\nLinear Regression model trained successfully.")


# ==========================================
# 22. REGRESSION EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_fare_test,
    y_fare_pred
)

rmse = mean_squared_error(
    y_fare_test,
    y_fare_pred
) ** 0.5

r2 = r2_score(
    y_fare_test,
    y_fare_pred
)

n = X_fare_test_processed.shape[0]
p = X_fare_test_processed.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\n========== REGRESSION EVALUATION ==========")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Adjusted R²: {adjusted_r2:.4f}")


# ==========================================
# 23. ACTUAL VS PREDICTED
# ==========================================

actual_vs_predicted = pd.DataFrame({
    "Actual Fare": y_fare_test.iloc[:10].values,
    "Predicted Fare": y_fare_pred[:10]
})

print("\n========== ACTUAL VS PREDICTED FARES ==========")
print(actual_vs_predicted.round(2))


# ==========================================
# 24. RESIDUAL ANALYSIS
# ==========================================

residuals = y_fare_test - y_fare_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_fare_pred, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")
plt.show()

print("""
Residual interpretation:
The residuals show that prediction errors are not perfectly random
around zero. Several observations have comparatively large errors,
especially at higher fare values. This indicates that the linear
model does not capture every relationship affecting fare.
""")


# ==========================================
# 25. SAVE / RELOAD FARE PIPELINE
# ==========================================

fare_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

fare_pipeline.fit(
    X_fare_train,
    y_fare_train
)

joblib.dump(
    fare_pipeline,
    "fare_regression_pipeline.joblib"
)

print("\nFare regression pipeline saved.")

loaded_fare_pipeline = joblib.load(
    "fare_regression_pipeline.joblib"
)

print("Fare regression pipeline reloaded successfully.")


# ==========================================
# 26. TEST FARE PIPELINE WITH RAW INPUT
# ==========================================

raw_passenger_fare = pd.DataFrame([{
    "age": 22,
    "sex": "male",
    "pclass": 3,
    "sibsp": 1,
    "parch": 0,
    "embarked": "S"
}])

fare_prediction = loaded_fare_pipeline.predict(
    raw_passenger_fare
)

print("\nRaw passenger for Fare prediction:")
print(raw_passenger_fare)

print("\nPredicted Fare:")
print(fare_prediction[0])


# ==========================================
# 27. FINAL CLASSIFICATION + REGRESSION TABLE
# ==========================================

classification_final_table = classification_comparison.copy()

classification_final_table["Task"] = "Classification"

classification_final_table["MAE"] = float("nan")
classification_final_table["RMSE"] = float("nan")
classification_final_table["R²"] = float("nan")
classification_final_table["Adjusted R²"] = float("nan")

regression_row = pd.DataFrame([{
    "Model": "Linear Regression",
    "Accuracy": float("nan"),
    "Precision": float("nan"),
    "Recall": float("nan"),
    "F1 Score": float("nan"),
    "AUC": float("nan"),
    "Task": "Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R²": r2,
    "Adjusted R²": adjusted_r2
}])

final_comparison = pd.concat(
    [classification_final_table, regression_row],
    ignore_index=True
)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    final_comparison[
        [
            "Task",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC",
            "MAE",
            "RMSE",
            "R²",
            "Adjusted R²"
        ]
    ].round(4).to_string(index=False)
)


# ==========================================
# 28. FINAL WRITTEN RECOMMENDATION
# ==========================================

print("\n========== FINAL WRITTEN RECOMMENDATION ==========")

print(f"""
For the Titanic classification task, the tested models were compared
using Accuracy, Precision, Recall, F1 Score and AUC. The stated
selection rule uses F1 Score first and AUC as a tie-breaker, resulting
in {selected_model_name} for the saved classification pipeline.

For the Fare regression task, Linear Regression was evaluated using
MAE, RMSE, R² and Adjusted R². The residual analysis indicates that
the model leaves some substantial prediction errors, particularly
for higher fare observations. These results should be considered
when interpreting the regression predictions.
""")


# ==========================================
# MODULE 2 COMPLETE
# ==========================================

print("\n==========================================")
print("MODULE 2 MACHINE LEARNING COMPLETE")
print("==========================================")
