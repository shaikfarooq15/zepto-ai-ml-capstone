import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "titanic.csv"

# =========================================================
# MODULE 2 - PART A: PROFILING, CLEANING AND DATA STORY
# =========================================================

# Load the raw dataset exactly once.
df = sns.load_dataset("titanic")

# Required offline fallback.
df.to_csv(CSV_PATH, index=False)

print(df.head())
print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nDataset info:")
df.info()

print("\nDataset description:")
print(df.describe(include="all"))

print("\nMissing values:")
print(df.isnull().sum())

missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)
print("\nMissing value percentages:")
print(missing_percentage[missing_percentage > 0])

print(f"\nOffline fallback saved to: {CSV_PATH}")


# =========================================================
# MISSING-VALUE DECISIONS
# =========================================================

df_clean = df.copy()

print("\n========== MISSING-VALUE STRATEGIES ==========")

for column in df_clean.columns:
    pct = df_clean[column].isnull().mean() * 100
    if pct > 0:
        print(f"{column}: {pct:.2f}% missing")

# Very high missingness -> drop column.
deck_pct = df_clean["deck"].isnull().mean() * 100
print(f"\ndeck: {deck_pct:.2f}% -> drop column because missingness is too high.")
df_clean = df_clean.drop(columns=["deck"])

# Under 5% -> drop affected rows.
for column in ["embarked", "embark_town"]:
    pct = df_clean[column].isnull().mean() * 100
    print(f"{column}: {pct:.2f}% -> drop affected rows because missingness is under 5%.")

df_clean = df_clean.dropna(subset=["embarked", "embark_town"])

# 5%-30% -> median imputation for the EDA cleaned DataFrame.
age_pct = df_clean["age"].isnull().mean() * 100
print(f"age: {age_pct:.2f}% -> median imputation because missingness is between 5% and 30%.")
df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

print("\nMissing values after EDA cleaning:")
print(df_clean.isnull().sum())
print("\nCleaned dataset shape:")
print(df_clean.shape)


# =========================================================
# UNIVARIATE ANALYSIS
# =========================================================

# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["age"], bins=20, kde=True)
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

print("\nAge interpretation:")
print("Most passengers were young to middle-aged adults, with a strong concentration around the 20-30 range.")
print("Older passengers were less common in the dataset.")

# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df_clean["fare"], bins=30, kde=True)
plt.title("Fare Distribution of Titanic Passengers")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Age box plot
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_clean["age"])
plt.title("Age Box Plot")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# Fare box plot
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_clean["fare"])
plt.title("Fare Box Plot")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()

# IQR outlier counts
for column in ["age", "fare"]:
    q1 = df_clean[column].quantile(0.25)
    q3 = df_clean[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df_clean[column] < lower) | (df_clean[column] > upper)).sum()
    print(f"\n{column.upper()} IQR OUTLIER CHECK")
    print(f"Q1: {q1:.4f}")
    print(f"Q3: {q3:.4f}")
    print(f"IQR: {iqr:.4f}")
    print(f"Lower bound: {lower:.4f}")
    print(f"Upper bound: {upper:.4f}")
    print(f"Outlier count: {count}")

fare_mean = df_clean["fare"].mean()
fare_median = df_clean["fare"].median()
fare_mode = df_clean["fare"].mode().iloc[0]

print("\n========== FARE STATISTICS ==========")
print(f"Mean: {fare_mean:.4f}")
print(f"Median: {fare_median:.4f}")
print(f"Mode: {fare_mode:.4f}")

if fare_mean > fare_median > fare_mode:
    print("Skewness conclusion: Fare is right-skewed because mean > median > mode.")
elif fare_mean < fare_median < fare_mode:
    print("Skewness conclusion: Fare is left-skewed because mean < median < mode.")
else:
    print("Skewness conclusion: The mean/median/mode ordering is not strictly monotonic; inspect the distribution plot.")


# =========================================================
# BIVARIATE ANALYSIS
# =========================================================

survival_by_sex = df_clean.groupby("sex")["survived"].mean()
print("\n========== SURVIVAL RATE BY SEX ==========")
print(survival_by_sex)

plt.figure(figsize=(7, 5))
sns.barplot(x=survival_by_sex.index, y=survival_by_sex.values)
plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

survival_by_class = df_clean.groupby("pclass")["survived"].mean()
print("\n========== SURVIVAL RATE BY PCLASS ==========")
print(survival_by_class)

plt.figure(figsize=(7, 5))
sns.barplot(x=survival_by_class.index.astype(str), y=survival_by_class.values)
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()

survival_by_sex_class = df_clean.groupby(["sex", "pclass"])["survived"].mean()
print("\n========== SURVIVAL RATE BY SEX + PCLASS ==========")
print(survival_by_sex_class)

# Exact six-column correlation matrix required by the specification.
corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df_clean[corr_columns].corr()

print("\n========== EXACT 6-COLUMN CORRELATION MATRIX ==========")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Titanic Correlation Matrix - Required Six Columns")
plt.tight_layout()
plt.show()

pairs = []
for i in range(len(corr_columns)):
    for j in range(i + 1, len(corr_columns)):
        a = corr_columns[i]
        b = corr_columns[j]
        value = corr_matrix.loc[a, b]
        pairs.append((a, b, value, abs(value)))

pairs.sort(key=lambda item: item[3], reverse=True)

print("\n========== TWO STRONGEST ABSOLUTE CORRELATIONS ==========")
for a, b, value, absolute_value in pairs[:2]:
    print(f"{a} vs {b}: correlation = {value:.4f}, absolute = {absolute_value:.4f}")


# =========================================================
# MULTIVARIATE DATA STORY - 4 DISTINCT CHARTS
# =========================================================

# Chart 1
plt.figure(figsize=(8, 5))
sns.barplot(data=df_clean, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Passenger Class and Sex")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.show()
print("\nChart 1 interpretation:")
print("Survival rates vary across both passenger class and sex.")
print("The combined chart shows why both variables should be considered when explaining survival rather than examining either variable alone.")

# Chart 2
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="survived", y="age")
plt.title("Age Distribution by Survival")
plt.xlabel("Survived")
plt.tight_layout()
plt.show()
print("\nChart 2 interpretation:")
print("The age distributions of survivors and non-survivors overlap substantially.")
print("This suggests age contributes information but does not by itself explain the survival outcome.")

# Chart 3
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x="pclass", y="fare", hue="survived")
plt.title("Fare Distribution by Class and Survival")
plt.tight_layout()
plt.show()
print("\nChart 3 interpretation:")
print("Fare varies substantially across passenger classes and also differs between survival groups within classes.")
print("This provides a multivariate view of the relationship between economic class, fare and survival.")

# Chart 4
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x="age", y="fare", hue="survived", style="sex", alpha=0.7)
plt.title("Age vs Fare by Survival and Sex")
plt.tight_layout()
plt.show()
print("\nChart 4 interpretation:")
print("The scatter plot combines age, fare, sex and survival in one view.")
print("The different combinations show that the outcome is influenced by multiple features rather than a single simple relationship.")


# =========================================================
# EXPLORATORY STANDARDIZATION CHECK
# =========================================================

standardization_data = df_clean[["age", "fare"]].copy()
scaler = StandardScaler()
standardized = pd.DataFrame(
    scaler.fit_transform(standardization_data),
    columns=["age", "fare"]
)

print("\n========== BEFORE STANDARDIZATION ==========")
print(standardization_data.agg(["mean", "std"]))

print("\n========== AFTER STANDARDIZATION ==========")
print(standardized.agg(["mean", "std"]))

print("\nStandardization interpretation:")
print("After z-score standardization, age and fare have approximately mean 0 and standard deviation 1.")
print("This is an EDA sanity check only; the modeling pipeline performs its own train-only preprocessing.")

print("\n==========================================")
print("TITANIC EDA COMPLETE")
print("==========================================")
