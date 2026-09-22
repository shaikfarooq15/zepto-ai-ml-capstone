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

Module 2 will build a complete Machine Learning workflow using the Titanic dataset.

The planned workflow includes:

```text
START
  ↓
Load Dataset
  ↓
Understand Data
  ↓
EDA
  ↓
Handle Missing Values
  ↓
Feature Engineering
  ↓
Train/Test Split
  ↓
Preprocessing
  ↓
Encoding
  ↓
Scaling
  ↓
Model Training
  ↓
Logistic Regression
  ↓
Decision Tree
  ↓
Random Forest
  ↓
Model Evaluation
  ↓
Class Imbalance
  ↓
SMOTE
  ↓
Hyperparameter Tuning
  ↓
Best Model
  ↓
Save Pipeline
  ↓
END
```

Module 2 will also include a regression task for predicting `Fare`.

---

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