# ==========================================
# MODULE 2 - MACHINE LEARNING
# Titanic Classification + Fare Regression
# ==========================================

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
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
# 1. LOAD AND PREPARE TITANIC DATA
# ==========================================

df = sns.load_dataset("titanic")

df_clean = df.copy()

# Drop very high-missing-value column
df_clean = df_clean.drop(columns=["deck"])

# Drop rows with very low missing percentage
df_clean = df_clean.dropna(
    subset=["embarked", "embark_town"]
)

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

print("Features:")
print(X.head())

print("\nTarget:")
print(y.head())


# ==========================================
# 2. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# ==========================================
# 3. PREPROCESSING
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

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training rules
X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)


# ==========================================
# 4. LOGISTIC REGRESSION
# ==========================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train_processed, y_train)

y_pred = model.predict(X_test_processed)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

y_probability = model.predict_proba(X_test_processed)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== LOGISTIC REGRESSION ==========")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nAUC:", auc_score)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.show()


# ==========================================
# 5. DECISION TREE
# ==========================================

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train_processed,
    y_train
)

tree_pred = decision_tree.predict(
    X_test_processed
)

tree_accuracy = accuracy_score(
    y_test,
    tree_pred
)

tree_precision = precision_score(
    y_test,
    tree_pred
)

tree_recall = recall_score(
    y_test,
    tree_pred
)

tree_f1 = f1_score(
    y_test,
    tree_pred
)

tree_cm = confusion_matrix(
    y_test,
    tree_pred
)

tree_probability = decision_tree.predict_proba(
    X_test_processed
)[:, 1]

tree_fpr, tree_tpr, tree_thresholds = roc_curve(
    y_test,
    tree_probability
)

tree_auc = roc_auc_score(
    y_test,
    tree_probability
)

print("\n========== DECISION TREE ==========")

print("Accuracy:", tree_accuracy)
print("Precision:", tree_precision)
print("Recall:", tree_recall)
print("F1 Score:", tree_f1)

print("\nConfusion Matrix:")
print(tree_cm)

print("\nAUC:", tree_auc)

plt.figure(figsize=(8, 5))
plt.plot(tree_fpr, tree_tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Decision Tree")
plt.show()


# ==========================================
# 6. RANDOM FOREST
# ==========================================

random_forest = RandomForestClassifier(
    random_state=42
)

random_forest.fit(
    X_train_processed,
    y_train
)

forest_pred = random_forest.predict(
    X_test_processed
)

forest_accuracy = accuracy_score(
    y_test,
    forest_pred
)

forest_precision = precision_score(
    y_test,
    forest_pred
)

forest_recall = recall_score(
    y_test,
    forest_pred
)

forest_f1 = f1_score(
    y_test,
    forest_pred
)

forest_cm = confusion_matrix(
    y_test,
    forest_pred
)

forest_probability = random_forest.predict_proba(
    X_test_processed
)[:, 1]

forest_fpr, forest_tpr, forest_thresholds = roc_curve(
    y_test,
    forest_probability
)

forest_auc = roc_auc_score(
    y_test,
    forest_probability
)

print("\n========== RANDOM FOREST ==========")

print("Accuracy:", forest_accuracy)
print("Precision:", forest_precision)
print("Recall:", forest_recall)
print("F1 Score:", forest_f1)

print("\nConfusion Matrix:")
print(forest_cm)

print("\nAUC:", forest_auc)

plt.figure(figsize=(8, 5))
plt.plot(forest_fpr, forest_tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.show()


# ==========================================
# 7. CLASS IMBALANCE
# ==========================================

print("\n========== CLASS DISTRIBUTION ==========")
print(y_train.value_counts())


# ------------------------------------------
# Balanced Logistic Regression
# ------------------------------------------

balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(
    X_train_processed,
    y_train
)

balanced_pred = balanced_model.predict(
    X_test_processed
)

balanced_accuracy = accuracy_score(
    y_test,
    balanced_pred
)

balanced_precision = precision_score(
    y_test,
    balanced_pred
)

balanced_recall = recall_score(
    y_test,
    balanced_pred
)

balanced_f1 = f1_score(
    y_test,
    balanced_pred
)

balanced_cm = confusion_matrix(
    y_test,
    balanced_pred
)

print("\n========== BALANCED LOGISTIC REGRESSION ==========")

print("Accuracy:", balanced_accuracy)
print("Precision:", balanced_precision)
print("Recall:", balanced_recall)
print("F1 Score:", balanced_f1)

print("\nConfusion Matrix:")
print(balanced_cm)


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


smote_model = LogisticRegression(
    max_iter=1000
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_pred = smote_model.predict(
    X_test_processed
)

smote_accuracy = accuracy_score(
    y_test,
    smote_pred
)

smote_precision = precision_score(
    y_test,
    smote_pred
)

smote_recall = recall_score(
    y_test,
    smote_pred
)

smote_f1 = f1_score(
    y_test,
    smote_pred
)

smote_cm = confusion_matrix(
    y_test,
    smote_pred
)

print("\n========== SMOTE LOGISTIC REGRESSION ==========")

print("Accuracy:", smote_accuracy)
print("Precision:", smote_precision)
print("Recall:", smote_recall)
print("F1 Score:", smote_f1)

print("\nConfusion Matrix:")
print(smote_cm)


# ==========================================
# 8. GRIDSEARCHCV - RANDOM FOREST
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

grid_search.fit(
    X_train_processed,
    y_train
)

print("\n========== GRID SEARCH ==========")

print("Best Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

print("\nBest Model OOB Score:")
print(best_model.oob_score_)


# ==========================================
# 9. TUNED RANDOM FOREST EVALUATION
# ==========================================

best_pred = best_model.predict(
    X_test_processed
)

best_accuracy = accuracy_score(
    y_test,
    best_pred
)

best_precision = precision_score(
    y_test,
    best_pred
)

best_recall = recall_score(
    y_test,
    best_pred
)

best_f1 = f1_score(
    y_test,
    best_pred
)

best_cm = confusion_matrix(
    y_test,
    best_pred
)

best_probability = best_model.predict_proba(
    X_test_processed
)[:, 1]

best_fpr, best_tpr, best_thresholds = roc_curve(
    y_test,
    best_probability
)

best_auc = roc_auc_score(
    y_test,
    best_probability
)

print("\n========== TUNED RANDOM FOREST ==========")

print("Accuracy:", best_accuracy)
print("Precision:", best_precision)
print("Recall:", best_recall)
print("F1 Score:", best_f1)

print("\nConfusion Matrix:")
print(best_cm)

print("\nAUC:", best_auc)


# ==========================================
# 10. MODEL COMPARISON
# ==========================================

model_comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Tuned Random Forest"
    ],
    "Accuracy": [
        accuracy,
        tree_accuracy,
        forest_accuracy,
        best_accuracy
    ],
    "Precision": [
        precision,
        tree_precision,
        forest_precision,
        best_precision
    ],
    "Recall": [
        recall,
        tree_recall,
        forest_recall,
        best_recall
    ],
    "F1 Score": [
        f1,
        tree_f1,
        forest_f1,
        best_f1
    ],
    "AUC": [
        auc_score,
        tree_auc,
        forest_auc,
        best_auc
    ]
})

print("\n========== MODEL COMPARISON ==========")
print(model_comparison.round(4))


# ==========================================
# 11. FINAL MODEL SELECTION
# ==========================================

print("\n========== FINAL MODEL SELECTION ==========")

print("""
Logistic Regression achieved:
Accuracy  = 80.90%
Precision = 78.33%
Recall    = 69.12%
F1 Score  = 73.44%
AUC       = 86.10%

Tuned Random Forest achieved:
Accuracy  = 80.90%
Precision = 78.33%
Recall    = 69.12%
F1 Score  = 73.44%
AUC       = 82.45%

Both models produced the same accuracy, precision, recall,
and F1 score on the test set.

Logistic Regression had the higher AUC in this evaluation.

Therefore, Logistic Regression is used as the final
classification model in the saved deployment pipeline.
""")


# ==========================================
# 12. FINAL CLASSIFICATION PIPELINE
# ==========================================

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Train complete pipeline using raw training data
final_pipeline.fit(
    X_train,
    y_train
)

print("\nFinal classification pipeline trained successfully!")


# Save classification pipeline
joblib.dump(
    final_pipeline,
    "titanic_survival_pipeline.joblib"
)

print("\nClassification pipeline saved successfully!")


# Reload classification pipeline
loaded_pipeline = joblib.load(
    "titanic_survival_pipeline.joblib"
)

print("\nClassification pipeline loaded successfully!")


# Test classification pipeline with raw input
raw_passenger = pd.DataFrame([{
    "age": 22,
    "fare": 7.25,
    "sex": "male",
    "pclass": 3,
    "sibsp": 1,
    "parch": 0,
    "embarked": "S"
}])

prediction = loaded_pipeline.predict(
    raw_passenger
)

print("\nRaw passenger:")
print(raw_passenger)

print("\nPredicted survival:")
print(prediction[0])


# ==========================================
# 13. FARE REGRESSION
# ==========================================

# Target: Fare
y_fare = df_clean["fare"]

# Fare is NOT included in the features
regression_features = [
    "age",
    "sex",
    "pclass",
    "sibsp",
    "parch",
    "embarked"
]

X_fare = df_clean[regression_features]

print("\n========== FARE REGRESSION ==========")

print("Regression X shape:")
print(X_fare.shape)

print("Regression y shape:")
print(y_fare.shape)

print("Regression features:")
print(regression_features)


# ==========================================
# 14. REGRESSION TRAIN / TEST SPLIT
# ==========================================

X_fare_train, X_fare_test, y_fare_train, y_fare_test = train_test_split(
    X_fare,
    y_fare,
    test_size=0.2,
    random_state=42
)

print("\nRegression train/test split:")
print("X_train:", X_fare_train.shape)
print("X_test:", X_fare_test.shape)
print("y_train:", y_fare_train.shape)
print("y_test:", y_fare_test.shape)


# ==========================================
# 15. REGRESSION PREPROCESSING
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


# Fit only on training data
X_fare_train_processed = reg_preprocessor.fit_transform(
    X_fare_train
)

# Transform test data
X_fare_test_processed = reg_preprocessor.transform(
    X_fare_test
)

print("\nProcessed regression data:")
print("X_train processed:", X_fare_train_processed.shape)
print("X_test processed:", X_fare_test_processed.shape)


# ==========================================
# 16. MULTIVARIATE LINEAR REGRESSION
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_fare_train_processed,
    y_fare_train
)

print("\nLinear Regression model trained successfully!")


# ==========================================
# 17. FARE PREDICTIONS
# ==========================================

y_fare_pred = linear_model.predict(
    X_fare_test_processed
)

print("\nFirst 10 actual vs predicted fares:")

print(
    pd.DataFrame({
        "Actual Fare": y_fare_test.iloc[:10].values,
        "Predicted Fare": y_fare_pred[:10]
    })
)


# ==========================================
# 18. REGRESSION EVALUATION
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

print("\n========== REGRESSION EVALUATION ==========")

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")


# Adjusted R²
n = X_fare_test_processed.shape[0]
p = X_fare_test_processed.shape[1]

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print(f"Adjusted R²: {adjusted_r2:.4f}")


# ==========================================
# 19. REGRESSION INTERPRETATION
# ==========================================

print("""
Regression Metric Interpretation:

MAE = 21.14 means the model's predictions are off by
about 21.14 fare units on average.

RMSE = 41.75 is higher than MAE because some predictions
have relatively large errors.

R² = 0.3468 means the model explains about 34.68%
of the variation in Fare.

Adjusted R² = 0.3118 accounts for the number of
predictors used by the regression model.
""")


# ==========================================
# 20. RESIDUAL PLOT
# ==========================================

residuals = y_fare_test - y_fare_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_fare_pred,
    residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot - Fare Regression")

plt.show()


print("""
Residual Plot Interpretation:

The residuals are not completely randomly distributed
around zero.

There are some visible patterns and several large positive
residuals, especially for higher predicted fares.

This shows that the linear regression model does not capture
all relationships in the Fare data and that some observations
have substantially larger prediction errors.
""")


# ==========================================
# 21. SAVE FARE REGRESSION PIPELINE
# ==========================================

fare_pipeline = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

# Train complete regression pipeline on raw training data
fare_pipeline.fit(
    X_fare_train,
    y_fare_train
)

# Save regression pipeline
joblib.dump(
    fare_pipeline,
    "fare_regression_pipeline.joblib"
)

print("\nFare regression pipeline saved successfully!")


# ==========================================
# 22. RELOAD FARE REGRESSION PIPELINE
# ==========================================

loaded_fare_pipeline = joblib.load(
    "fare_regression_pipeline.joblib"
)

print("\nFare regression pipeline loaded successfully!")


# ==========================================
# 23. TEST FARE PIPELINE WITH RAW INPUT
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
# MODULE 2 COMPLETE
# ==========================================

print("\n==========================================")
print("MODULE 2 MACHINE LEARNING COMPLETE")
print("==========================================")