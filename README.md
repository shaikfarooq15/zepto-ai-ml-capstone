# Zepto AI/ML Capstone Project

An end-to-end AI/ML engineering project covering:

- Data Collection
- Data Cleaning
- SQL Database
- Machine Learning
- Generative AI
- FastAPI
- Docker

---

# Module 1 — Data Pipeline

## 1. Project Objective

The goal of Module 1 is to build a complete data pipeline using data collected from:

https://books.toscrape.com/

The pipeline performs:

1. Web scraping
2. Data cleaning
3. Pandas processing
4. Currency conversion
5. SQLite database creation
6. SQL analysis
7. Pandas SQL analysis

---

## 2. Data Collection

The website was scraped using:

- Python
- Requests
- BeautifulSoup

Four book categories were collected:

- Travel
- Mystery
- Historical Fiction
- Fiction

A total of **71 books** were collected.

Each book contains:

- Title
- Price
- Star Rating
- Availability
- Category

---

## 3. Data Cleaning

The raw website values were converted into clean values that can be used for data analysis.

### Price

Raw value:

```text
£45.17
```

Clean value:

```text
45.17
```

The pound symbol was removed and the value was converted to a numeric `float`.

The cleaned column is:

```text
price_gbp
```

### Rating

Raw value:

```text
Four
```

The star rating was converted into a numeric value:

```text
One   → 1
Two   → 2
Three → 3
Four  → 4
Five  → 5
```

The cleaned column is:

```text
rating
```

### Availability

Raw value:

```text
In stock
```

The value was converted into a Boolean value:

```text
True
```

The cleaned column is:

```text
in_stock
```

This makes the data easier to filter and analyze.

---

## 4. Pandas DataFrame

After scraping and cleaning the data, the collected books were converted into a Pandas DataFrame.

The main columns are:

```text
title
price_gbp
rating
in_stock
category
```

The data types were checked using:

```python
df.dtypes
```

The resulting data contains:

- `title` → string
- `price_gbp` → numeric
- `rating` → integer
- `in_stock` → Boolean
- `category` → string

---

## 5. GBP to INR Currency Conversion

The project also converts the book prices from GBP to INR.

A fixed conversion rate of:

```text
1 GBP = 115 INR
```

was used for this project.

The conversion was performed using:

```python
GBP_TO_INR = 115

df["price_inr"] = df["price_gbp"] * GBP_TO_INR
```

For example:

```text
45.17 GBP × 115 = 5194.55 INR
```

The final dataset therefore contains both:

```text
price_gbp
price_inr
```

---

## 6. SQLite Database

The cleaned data was stored in a SQLite database.

Database location:

```text
data/books.db
```

The database contains two related tables:

```text
Categories
Books
```

---

## 7. Database Relationship

The two tables are connected using a primary key and foreign key.

### Categories Table

The `Categories` table contains:

```text
category_id
category_name
```

`category_id` is the primary key.

### Books Table

The `Books` table contains:

```text
book_id
title
price_gbp
price_inr
rating
in_stock
category_id
```

`book_id` is the primary key.

`category_id` is a foreign key that connects each book to its category.

---

## 8. Database Relationship Diagram

```text
Categories
    |
    | category_id (PK)
    |
    | 1-to-many
    ↓
Books
    |
    | category_id (FK)
    ↓
Book Records
```

One category can contain many books.

Example:

```text
Travel
   |
   ├── Book 1
   ├── Book 2
   ├── Book 3
   └── ...
```

---

## 9. SQL Analysis

SQL was used to analyze the data stored inside the SQLite database.

The project contains more than five SQL queries covering:

- SELECT
- WHERE
- ORDER BY
- LIMIT
- DISTINCT
- IN
- BETWEEN
- JOIN
- GROUP BY
- COUNT

---

## Query 1 — Books Above £40

```sql
SELECT title, price_gbp
FROM Books
WHERE price_gbp > 40
ORDER BY price_gbp DESC
LIMIT 10;
```

This query finds books costing more than £40 and sorts them from the highest price to the lowest price.

`LIMIT 10` returns only the first 10 results.

---

## Query 2 — DISTINCT and IN

```sql
SELECT DISTINCT category_id
FROM Books
WHERE category_id IN (1, 2, 3);
```

This query demonstrates:

- `DISTINCT`
- `IN`

It returns unique category IDs from the selected categories.

---

## Query 3 — BETWEEN

```sql
SELECT title, price_gbp
FROM Books
WHERE price_gbp BETWEEN 20 AND 30
ORDER BY price_gbp;
```

This query finds books whose prices are between £20 and £30.

The results are ordered from the lowest price to the highest price.

---

## Query 4 — JOIN Between Books and Categories

```sql
SELECT
    Books.title,
    Books.price_gbp,
    Categories.category_name
FROM Books
JOIN Categories
    ON Books.category_id = Categories.category_id
LIMIT 10;
```

This query combines information from both tables.

The `Books` table contains the `category_id`, while the actual category name is stored in the `Categories` table.

The JOIN connects them using:

```text
Books.category_id
        =
Categories.category_id
```

This allows us to see:

```text
Book Title
Price
Category Name
```

together.

---

## Query 5 — Number of Books in Each Category

```sql
SELECT
    Categories.category_name,
    COUNT(Books.book_id) AS book_count
FROM Categories
JOIN Books
    ON Categories.category_id = Books.category_id
GROUP BY Categories.category_name
ORDER BY book_count DESC;
```

The result was:

```text
Mystery             20
Historical Fiction  20
Fiction             20
Travel              11
```

Total:

```text
71 books
```

---

## 10. SQL Results with Pandas

SQL results were also loaded into Pandas using:

```python
pandas.read_sql()
```

Example:

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("data/books.db")

query = """
SELECT
    Categories.category_name,
    COUNT(Books.book_id) AS book_count
FROM Categories
JOIN Books
    ON Categories.category_id = Books.category_id
GROUP BY Categories.category_name
ORDER BY book_count DESC
"""

df = pd.read_sql(query, conn)

print(df)

conn.close()
```

Output:

```text
       category_name  book_count
0            Mystery          20
1  Historical Fiction          20
2            Fiction          20
3            Travel          11
```

This demonstrates how SQL and Pandas can work together in a data pipeline.

---

## 11. Reproducing SQL JOIN Using Pandas

The SQL JOIN was also reproduced using `pandas.merge()`.

First, both tables were loaded into Pandas:

```python
books_df = pd.read_sql("SELECT * FROM Books", conn)

categories_df = pd.read_sql(
    "SELECT * FROM Categories",
    conn
)
```

Then the tables were merged:

```python
merged_df = pd.merge(
    books_df,
    categories_df,
    on="category_id",
    how="inner"
)
```

The final result contains information such as:

```text
title
price_gbp
category_name
```

This demonstrates that the same relational operation can be performed using Pandas.

---

## 12. Project Structure

```text
zepto-ai-ml-capstone/
│
├── data/
│   └── books.db
│
├── module1_data_pipeline/
│   ├── scraper.py
│   └── check_database.py
│
├── module2_machine_learning/
│
├── module3_genai_assistant/
│
├── tests/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 13. Module 1 Technologies

The following technologies were used in Module 1:

- Python
- Requests
- BeautifulSoup
- Pandas
- SQLite
- SQL

---

## 14. Module 1 Pipeline

The complete Module 1 workflow is:

```text
START
  ↓
books.toscrape.com
  ↓
Web Scraping
  ↓
Requests + BeautifulSoup
  ↓
Raw Book Data
  ↓
Data Cleaning
  ↓
Pandas DataFrame
  ↓
GBP → INR Conversion
  ↓
SQLite Database
  ↓
Categories + Books Tables
  ↓
SQL Queries
  ↓
JOIN
  ↓
Pandas read_sql()
  ↓
Pandas merge()
  ↓
Analysis Results
  ↓
END
```

---

## 15. Module 1 Final Checklist

### Data Collection

- [x] Scraped books from books.toscrape.com
- [x] Collected 71 books
- [x] Collected 4 categories
- [x] Collected Title
- [x] Collected Price
- [x] Collected Star Rating
- [x] Collected Availability
- [x] Collected Category

### Data Cleaning

- [x] Converted price to numeric GBP
- [x] Converted ratings to numeric values
- [x] Converted availability to Boolean
- [x] Created Pandas DataFrame
- [x] Added INR price column

### Database

- [x] Created SQLite database
- [x] Created Categories table
- [x] Created Books table
- [x] Added primary keys
- [x] Added foreign key relationship
- [x] Inserted cleaned data

### SQL

- [x] SELECT
- [x] WHERE
- [x] ORDER BY
- [x] LIMIT
- [x] DISTINCT
- [x] IN
- [x] BETWEEN
- [x] JOIN
- [x] GROUP BY
- [x] COUNT

### Pandas + SQL

- [x] Used `pandas.read_sql()`
- [x] Reproduced JOIN using `pandas.merge()`

---

## 16. Module 1 Status

Module 1 Data Pipeline is complete.

The pipeline successfully demonstrates:

```text
START
  ↓
Web Scraping
  ↓
Data Cleaning
  ↓
Pandas
  ↓
Currency Conversion
  ↓
SQLite
  ↓
SQL
  ↓
JOIN
  ↓
Pandas SQL Analysis
  ↓
END
```

---

# Module 2 — Machine Learning

Module 2 builds a complete Machine Learning workflow using the Titanic dataset.

The module contains two Machine Learning tasks:

1. Classification — predicting passenger survival
2. Regression — predicting passenger fare

---

## 1. Dataset

The Titanic dataset was loaded using Seaborn.

### Original Dataset

```text
Rows: 891
Columns: 15
```

### Classification Target

```text
survived
```

Where:

```text
0 → Did not survive
1 → Survived
```

---

## 2. Missing Value Analysis

Missing values were measured before selecting the cleaning strategy.

### Age

```text
Missing values: 177
Missing percentage: 19.87%
Strategy: Median imputation
```

### Embarked

```text
Missing values: 2
Missing percentage: 0.22%
Strategy: Drop rows
```

### Embark Town

```text
Missing values: 2
Missing percentage: 0.22%
Strategy: Drop rows
```

### Deck

```text
Missing values: 688
Missing percentage: 77.22%
Strategy: Drop column
```

### After Cleaning

```text
Rows: 889
Columns: 14
Missing values: 0
```

---

## 3. Exploratory Data Analysis

The following variables were explored:

- Age
- Fare
- Sex
- Passenger class
- Survival

### Visualizations

The project contains five main visualizations:

1. Age histogram
2. Fare box plot
3. Survival by Sex
4. Survival by Passenger Class
5. Correlation heatmap

### Age Histogram

The distribution shows that many passengers were in the young-to-middle-age range, with a strong concentration around approximately 20–30 years.

### Fare Box Plot

Most passengers paid relatively low fares, while some passengers paid substantially higher fares. The distribution contains high-fare outliers.

### Survival by Sex

Female passengers had a substantially higher survival rate than male passengers in this dataset.

### Survival by Passenger Class

Survival rates differed across passenger classes:

| Passenger Class | Approx. Survival Rate |
|---|---:|
| 1st Class | 63% |
| 2nd Class | 47% |
| 3rd Class | 24% |

These are associations observed in this dataset and do not by themselves establish causation.

### Correlation Heatmap

Among the numerical variables, passenger class and fare showed stronger linear relationships with survival than variables such as age, SibSp, and Parch.

> Correlation does not imply causation.

---

## 4. Feature Preparation

### Classification Features

```text
age
fare
sex
pclass
sibsp
parch
embarked
```

### Target

```text
survived
```

The dataset was divided using a stratified train/test split.

```text
Training data: 711 rows
Testing data: 178 rows
```

Stratification was used to preserve the target-class distribution between training and testing data.

---

## 5. Data Preprocessing

A `ColumnTransformer` was used to apply different preprocessing to numerical and categorical features.

### Numerical Features

```text
age
fare
pclass
sibsp
parch
```

### Numerical Processing

```text
Median Imputation
        ↓
StandardScaler
```

### Categorical Features

```text
sex
embarked
```

### Categorical Processing

```text
Most-Frequent Imputation
        ↓
OneHotEncoder
```

### Processed Data

```text
Training: 711 × 10
Testing: 178 × 10
```

### Preventing Data Leakage

Preprocessing was fitted only on the training data:

```python
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

This prevents information from the test set from being used while learning preprocessing parameters.

---

## 6. Classification Models

Three classification models were trained:

- Logistic Regression
- Decision Tree
- Random Forest

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC

---

## 7. Logistic Regression

### Results

| Metric | Result |
|---|---:|
| Accuracy | 80.90% |
| Precision | 78.33% |
| Recall | 69.12% |
| F1 Score | 73.44% |
| AUC | 86.10% |

### Confusion Matrix

```text
[[97 13]
 [21 47]]
```

The ROC/AUC evaluation produced an AUC of approximately 0.861 on the test set.

---

## 8. Decision Tree

### Results

| Metric | Result |
|---|---:|
| Accuracy | 76.97% |
| Precision | 69.01% |
| Recall | 72.06% |
| F1 Score | 70.50% |
| AUC | 75.41% |

### Confusion Matrix

```text
[[88 22]
 [19 49]]
```

---

## 9. Random Forest

### Results

| Metric | Result |
|---|---:|
| Accuracy | 80.34% |
| Precision | 77.05% |
| Recall | 69.12% |
| F1 Score | 72.87% |
| AUC | 82.25% |

### Confusion Matrix

```text
[[96 14]
 [21 47]]
```

---

## 10. Class Imbalance

The training target distribution was:

```text
0 → 439
1 → 272
```

The classes were therefore not evenly distributed.

Two balancing approaches were tested:

- `class_weight="balanced"`
- SMOTE

---

## 11. Balanced Logistic Regression

The Logistic Regression model was trained using:

```python
class_weight="balanced"
```

### Results

| Metric | Result |
|---|---:|
| Accuracy | 79.21% |
| Precision | 71.83% |
| Recall | 75.00% |
| F1 Score | 73.38% |

Compared with the baseline model, recall increased from:

```text
69.12% → 75.00%
```

while accuracy and precision decreased.

---

## 12. SMOTE

SMOTE was applied **only to the training data**.

### Before SMOTE

```text
0 → 439
1 → 272
```

### After SMOTE

```text
0 → 439
1 → 439
```

The test data remained untouched.

### SMOTE Logistic Regression Results

| Metric | Result |
|---|---:|
| Accuracy | 79.78% |
| Precision | 73.53% |
| Recall | 73.53% |
| F1 Score | 73.53% |

The experiment demonstrates that class-balancing methods can change the trade-off between recall, precision, and accuracy.

---

## 13. Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune the Random Forest.

### Parameters Searched

```text
n_estimators
max_depth
max_features
```

### Best Parameters

```text
max_depth = 10
max_features = sqrt
n_estimators = 50
```

### Best OOB Score

```text
81.01%
```

---

## 14. Tuned Random Forest

### Test-Set Results

| Metric | Result |
|---|---:|
| Accuracy | 80.90% |
| Precision | 78.33% |
| Recall | 69.12% |
| F1 Score | 73.44% |
| AUC | 82.45% |

### Confusion Matrix

```text
[[97 13]
 [21 47]]
```

---

## 15. Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.90% | 78.33% | 69.12% | 73.44% | 86.10% |
| Decision Tree | 76.97% | 69.01% | 72.06% | 70.50% | 75.41% |
| Random Forest | 80.34% | 77.05% | 69.12% | 72.87% | 82.25% |
| Tuned Random Forest | 80.90% | 78.33% | 69.12% | 73.44% | 82.45% |

The Logistic Regression and tuned Random Forest produced the same accuracy, precision, recall, and F1 score on this test set.

Logistic Regression had the higher AUC in this evaluation and was therefore used as the final classification model in the saved deployment pipeline.

---

## 16. Final Classification Pipeline

The complete classification workflow was saved as a single pipeline.

### Pipeline

```text
Raw Input
    ↓
Preprocessing
    ↓
Encoding
    ↓
Scaling
    ↓
Logistic Regression
    ↓
Prediction
```

### Saved Pipeline

```text
module2_machine_learning/titanic_survival_pipeline.joblib
```

The saved pipeline was:

1. Loaded again
2. Given raw passenger data
3. Used to generate a prediction

### Example Raw Passenger

```text
age       = 22
fare      = 7.25
sex       = male
pclass    = 3
sibsp     = 1
parch     = 0
embarked  = S
```

### Prediction

```text
0 → Did not survive
```

---

## 17. Fare Regression

A second Machine Learning task was implemented to predict passenger fare.

### Target

```text
fare
```

### Features

```text
age
sex
pclass
sibsp
parch
embarked
```

### Model

```text
Linear Regression
```

### Dataset

```text
889 rows
6 input features
```

### Train/Test Split

```text
Training: 711
Testing: 178
```

---

## 18. Regression Evaluation

The regression model was evaluated using:

- MAE
- RMSE
- R²
- Adjusted R²
- Residual Plot

### MAE

```text
21.14
```

The predictions were off by approximately 21.14 fare units on average.

### RMSE

```text
41.75
```

RMSE was higher than MAE because some observations had relatively large prediction errors.

### R²

```text
0.3468
```

The model explains approximately 34.68% of the variation in the Fare target.

### Adjusted R²

```text
0.3118
```

Adjusted R² accounts for the number of predictors used by the regression model.

---

## 19. Residual Analysis

A residual plot was generated to analyze the regression errors.

The residuals were not completely randomly distributed around zero.

There were visible patterns and several larger residuals, particularly around higher predicted fares.

This indicates that the linear regression model does not capture all relationships in the Fare data and that some observations have substantially larger prediction errors.

---

## 20. Fare Regression Pipeline

The complete Fare regression workflow was also saved using Joblib.

### Pipeline

```text
Raw Input
    ↓
Preprocessing
    ↓
Linear Regression
    ↓
Fare Prediction
```

### Saved Pipeline

```text
module2_machine_learning/fare_regression_pipeline.joblib
```

The pipeline was reloaded and tested using raw passenger data.

### Example

```text
age       = 22
sex       = male
pclass    = 3
sibsp     = 1
parch     = 0
embarked  = S
```

### Predicted Fare

```text
3.9721
```

---

## 21. Module 2 Project Structure

```text
module2_machine_learning/
│
├── model_pipeline.py
├── titanic_analysis.py
├── titanic_survival_pipeline.joblib
└── fare_regression_pipeline.joblib
```

---

## 22. Module 2 Technologies

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Joblib

---

## 23. Module 2 Complete Workflow

```text
START
  ↓
Load Titanic Dataset
  ↓
Understand Data
  ↓
Missing Value Analysis
  ↓
Data Cleaning
  ↓
EDA + Visualization
  ↓
Feature Selection
  ↓
Train / Test Split
  ↓
Preprocessing
  ↓
Encoding + Scaling
  ↓
Classification Models
  ↓
Model Evaluation
  ↓
Class Imbalance
  ↓
Balanced Logistic Regression
  ↓
SMOTE
  ↓
Random Forest
  ↓
GridSearchCV
  ↓
Model Comparison
  ↓
Final Classification Pipeline
  ↓
Joblib Save / Reload
  ↓
Raw Passenger Prediction
  ↓
Fare Regression
  ↓
Regression Evaluation
  ↓
Residual Analysis
  ↓
Fare Pipeline Save / Reload
  ↓
Raw Fare Prediction
  ↓
END
```

---

## 24. Module 2 Final Checklist

### Dataset & EDA

- [x] Loaded Titanic dataset using Seaborn
- [x] Checked dataset shape and structure
- [x] Measured missing values
- [x] Calculated missing percentages
- [x] Applied missing-value strategies
- [x] Explored Age
- [x] Explored Fare
- [x] Explored Sex
- [x] Explored Passenger Class
- [x] Explored Survival
- [x] Created histogram
- [x] Created box plot
- [x] Created survival comparison charts
- [x] Created correlation heatmap

### Classification

- [x] Created target variable
- [x] Created features
- [x] Train/test split
- [x] Preprocessing
- [x] Encoding
- [x] Scaling
- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1 Score
- [x] Confusion Matrix
- [x] ROC
- [x] AUC

### Class Imbalance

- [x] Checked class distribution
- [x] Baseline model
- [x] `class_weight="balanced"`
- [x] SMOTE
- [x] SMOTE applied only to training data
- [x] Compared balancing approaches

### Model Tuning

- [x] GridSearchCV
- [x] Tuned `n_estimators`
- [x] Tuned `max_depth`
- [x] Tuned `max_features`
- [x] OOB score
- [x] Compared tuned model with other models

### Regression

- [x] Fare prediction
- [x] Multivariate Linear Regression
- [x] MAE
- [x] RMSE
- [x] R²
- [x] Adjusted R²
- [x] Residual plot
- [x] Residual interpretation

### Deployment Preparation

- [x] Saved classification pipeline with Joblib
- [x] Reloaded classification pipeline
- [x] Tested raw passenger input
- [x] Saved Fare regression pipeline
- [x] Reloaded Fare regression pipeline
- [x] Tested raw passenger input

---

## 25. Module 2 Status

Module 2 Machine Learning is **complete**.

The module demonstrates an end-to-end Machine Learning workflow:

```text
DATA
  ↓
EDA
  ↓
CLEANING
  ↓
PREPROCESSING
  ↓
CLASSIFICATION
  ↓
EVALUATION
  ↓
IMBALANCE HANDLING
  ↓
HYPERPARAMETER TUNING
  ↓
MODEL SELECTION
  ↓
REGRESSION
  ↓
ERROR ANALYSIS
  ↓
JOBLIB PIPELINES
  ↓
RAW INPUT PREDICTION
```

# Module 3 — Generative AI Assistant

Module 3 will build a Zepto support assistant using policy documents.

The planned architecture is:

```text
START
  ↓
Zepto Policy Documents
  ↓
Document Loading
  ↓
Chunking
  ↓
Embeddings
  ↓
ChromaDB
  ↓
Semantic Retrieval
  ↓
Top 3 Relevant Chunks
  ↓
LangGraph
  ↓
Policy / General Intent
  ↓
Grounded Answer
  ↓
FastAPI
  ↓
Docker
  ↓
END
```

The assistant will support:

- Policy question answering
- Document retrieval
- Embeddings
- ChromaDB
- LangGraph
- Structured output
- FastAPI
- Docker

The required `MOCK_LLM` mode will allow the application to work without an external paid LLM API.

---

# Complete Capstone Architecture

The complete project connects three major stages:

```text
                         START
                           ↓
                  Zepto AI/ML CAPSTONE
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   MODULE 1           MODULE 2           MODULE 3
   DATA PIPELINE      MACHINE LEARNING   GENERATIVE AI
        ↓                  ↓                  ↓
   Web Scraping          Titanic          Policy Docs
        ↓                  ↓                  ↓
   Pandas                EDA               Chunking
        ↓                  ↓                  ↓
   SQLite                ML Models         Embeddings
        ↓                  ↓                  ↓
   SQL                   Evaluation        ChromaDB
        ↓                  ↓                  ↓
        └──────────────────┼──────────────────┘
                           ↓
                     FastAPI / Docker
                           ↓
                          END
```

---

# Conclusion

This project demonstrates an end-to-end AI/ML engineering workflow:

```text
START
  ↓
DATA
  ↓
CLEANING
  ↓
DATABASE
  ↓
SQL
  ↓
MACHINE LEARNING
  ↓
GENERATIVE AI
  ↓
API
  ↓
DOCKER
  ↓
END
```

The goal is to demonstrate practical understanding of the complete AI/ML development lifecycle rather than building isolated notebooks.

---

# Author

**Shaik Mohammad Umar Farooq**

B.Tech — Computer Science and Engineering (AI & ML)

Zepto Data & AI Platform — Certificate Program in Artificial Intelligence and Machine Learning