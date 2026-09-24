# Zepto AI/ML Capstone

This project brings together three connected parts of the Zepto AI/ML capstone:

1. Data pipeline: web scraping, cleaning, currency conversion, SQLite storage, and SQL analysis. See [data_pipeline/README.md](data_pipeline/README.md).
2. Analytics: Titanic data preparation, EDA, classification, imbalance handling, model tuning, and fare regression. See [analytics/README.md](analytics/README.md).
3. Support assistant: a local RAG-based Zepto policy assistant using embeddings, ChromaDB, LangGraph, and FastAPI. See [support_assistant/README.md](support_assistant/README.md).

All dependencies are listed in the root `requirements.txt` file.

## Setup

    pip install -r requirements.txt

Python 3.11+ is recommended.

The project uses relative paths, so it can be copied to another folder or computer.

The first run needs internet access for the BooksToScrape website and the Sentence Transformer embedding model.

---

## Module 1: Data Pipeline

Run from the project root:

    python data_pipeline/scraper.py

    python data_pipeline/check_database.py

The pipeline:

- Scrapes books from BooksToScrape
- Collects 71 books from 4 categories
- Cleans price, rating, and availability
- Converts GBP to INR using 1 GBP = 105.50 INR
- Creates the SQLite database
- Creates related Categories and Books tables
- Performs SQL analysis
- Reproduces SQL JOIN using Pandas

Database:

    data/books.db

---

## Module 2: Analytics

Run from the project root:

    python analytics/titanic_analysis.py

    python analytics/model_pipeline.py

The analytics pipeline:

- Cleans the Titanic dataset
- Performs EDA
- Analyzes missing values and outliers
- Standardizes numerical features
- Trains classification models
- Handles class imbalance using class weighting and SMOTE
- Tunes Random Forest using GridSearchCV
- Performs fare regression
- Saves and reloads ML pipelines

Classification models:

- Logistic Regression
- Decision Tree
- Random Forest
- Tuned Random Forest

Regression model:

    Linear Regression

---

## Module 3: Support Assistant

Run from the project root:

    cd support_assistant
    python -m uvicorn main:app --reload --port 8000

Open the API documentation:

    http://127.0.0.1:8000/docs

The assistant:

- Loads 8 Zepto policy documents
- Creates embeddings using Sentence Transformers
- Stores embeddings in ChromaDB
- Retrieves the top relevant documents
- Uses LangGraph for intent routing
- Returns structured JSON responses
- Exposes the `/ask` FastAPI endpoint

Example request:

    POST /ask

    {
        "query": "What is the delivery policy?"
    }

`MOCK_LLM` is enabled by default, so the application can run without a paid external LLM API.

---

## Docker

A Dockerfile is included for the Support Assistant.

Build:

    docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

Run:

    docker run --rm -p 7860:7860 zepto-support-assistant

The Docker service exposes:

    http://127.0.0.1:7860

Note: Docker was not available on the development machine during testing, so the Docker build itself was not executed locally.

---

## Notes on the Implementation

- Data Pipeline: Requests and BeautifulSoup are used for scraping, Pandas handles cleaning, and SQLite stores the normalized relational data.
- Analytics: the Titanic workflow includes EDA, classification, imbalance handling, Random Forest tuning, and fare regression. Joblib is used for pipeline persistence.
- Support Assistant: policy documents and vector storage remain local. Sentence Transformers provides embeddings and ChromaDB provides semantic retrieval.
- Support Assistant: LangGraph handles policy/general intent routing and FastAPI exposes the application.
- `MOCK_LLM` allows the Support Assistant to run without an external paid LLM service.

---

## Requirement Checklist

| Requirement | Status | Where to check |
|---|---|---|
| Three modules in the repository | Complete | `data_pipeline/`, `analytics/`, `support_assistant/` |
| One dependency file | Complete | `requirements.txt` |
| 71 books and 4 categories | Complete | `data_pipeline/scraper.py` |
| Data cleaning and currency conversion | Complete | `data_pipeline/scraper.py` |
| Related SQLite tables | Complete | `data/books.db` |
| SQL analysis | Complete | `data_pipeline/check_database.py` |
| Pandas SQL JOIN comparison | Complete | `data_pipeline/check_database.py` |
| Titanic EDA | Complete | `analytics/titanic_analysis.py` |
| Classification models | Complete | `analytics/model_pipeline.py` |
| Class imbalance handling | Complete | `analytics/model_pipeline.py` |
| Random Forest tuning | Complete | `analytics/model_pipeline.py` |
| Fare regression | Complete | `analytics/model_pipeline.py` |
| Saved classification pipeline | Complete | `titanic_survival_pipeline.joblib` |
| Saved regression pipeline | Complete | `fare_regression_pipeline.joblib` |
| 8 policy documents | Complete | `support_assistant/docs/` |
| Embeddings and ChromaDB | Complete | `support_assistant/main.py` |
| LangGraph routing | Complete | `support_assistant/main.py` |
| FastAPI `/ask` endpoint | Complete | `support_assistant/main.py` |
| Mock LLM mode | Complete | `MOCK_LLM` |
| Dockerfile | Complete | `support_assistant/Dockerfile` |

---

## Verification

### Data Pipeline

    python data_pipeline/scraper.py
    python data_pipeline/check_database.py

### Analytics

    python analytics/titanic_analysis.py
    python analytics/model_pipeline.py

### Support Assistant

    cd support_assistant
    python -m uvicorn main:app --reload --port 8000

Test the API through:

    http://127.0.0.1:8000/docs

---

## Project Status

- [x] Data Pipeline
- [x] Analytics
- [x] Support Assistant
- [x] Module READMEs
- [x] Root README
- [x] Requirements file
- [x] Dockerfile
- [x] API testing

The Zepto AI/ML Capstone is ready for final repository review and submission.