import seaborn as sns
import pandas as pd

df = sns.load_dataset("titanic")

print(df.head())
print("\nDataset shape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing value percentages:")
print(df.isnull().mean() * 100)
import matplotlib.pyplot as plt

plt.hist(df["age"].dropna(), bins=20)

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.show()

# Age Distribution:
# Most passengers were young to middle-aged adults.
# The largest concentration was around the 20-30 age range.
# Fewer passengers were present at older ages.

plt.boxplot(df["fare"])
plt.title("Fare Distribution of Titanic Passengers")
plt.ylabel("Fare")
plt.show()

# Fare Distribution:
# Most passengers paid relatively low fares.
# The fare distribution is strongly right-skewed.
# There are several high-fare outliers.

survival_by_sex = df.groupby("sex")["survived"].mean()

survival_by_sex.plot(kind="bar")

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.show()

# Survival Rate by Sex:
# Female passengers had a much higher observed survival rate
# than male passengers in this dataset.
# Female survival was around 74%, while male survival was around 19%.

survival_by_class = df.groupby("pclass")["survived"].mean()

survival_by_class.plot(kind="bar")

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# Survival Rate by Passenger Class:
# First-class passengers had the highest observed survival rate,
# followed by second-class and then third-class passengers.
# Survival rates were approximately 63%, 47%, and 24% respectively.

numeric_data = df.select_dtypes(include="number")

correlation = numeric_data.corr()

sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Correlation Heatmap:
# Passenger class and fare showed the strongest linear relationships
# with survival among the numerical variables examined.
# Age, sibsp, and parch showed relatively weak linear correlations
# with survival.

df_clean = df.copy()

df_clean = df_clean.drop(columns=["deck"])

# Deck: 77.216611% missing
# Strategy: Drop the column because the missing rate is extremely high
# and imputing such a large number of values would be unreliable.

df_clean = df_clean.dropna(subset=["embarked", "embark_town"])

# Embark_town: 0.224467% missing
# Strategy: Drop the affected rows because the missing rate is below 5%.

df_clean["age"] = df_clean["age"].fillna(df_clean["age"].median())

# Age: 19.865320% missing
# Strategy: Median imputation because the missing rate is between 5% and 30%.

print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

print("\nCleaned dataset shape:")
print(df_clean.shape)
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())