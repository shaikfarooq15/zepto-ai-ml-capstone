# Support Assistant

This module is a small Zepto policy assistant. It searches the local policy files with sentence-transformer embeddings, stores the vectors in ChromaDB, and uses LangGraph to choose how to answer. Mock mode is on by default, so no API key is needed.

## Setup

From the repo root:

    pip install -r requirements.txt
    cd support_assistant
    python -m uvicorn main:app --reload

Then call the local FastAPI app:

    POST http://127.0.0.1:8000/ask

The Swagger documentation is available at:

    http://127.0.0.1:8000/docs

The first request downloads all-MiniLM-L6-v2 if it is not already cached. The embeddings are stored in the local ChromaDB directory.

## How it works

The request follows this path:

    docs/*.txt
        -> load_documents()
        -> SentenceTransformer("all-MiniLM-L6-v2")
        -> ChromaDB collection: zepto_policies
        -> classify_intent node
           -> policy_question: retrieve_and_answer node -> JSON response
           -> general_question: direct_answer node -> JSON response

1. **Ingestion:** `load_documents()` reads the eight files under `support_assistant/docs`. Each document is loaded separately and assigned a document ID.

2. **Embedding:** `SentenceTransformer("all-MiniLM-L6-v2")` embeds each policy document and each incoming query locally.

3. **Vector Storage:** ChromaDB stores the policy documents, document IDs, and their embeddings in the `zepto_policies` collection.

4. **Retrieval:** The `retrieve_and_answer` node embeds policy questions and retrieves the top three matching documents from ChromaDB.

5. **Generation:** In the default mock mode, `retrieve_and_answer` returns the required `Based on the retrieved context: ...` response. The `direct_answer` node returns the fixed policy-only response.

The graph is built using `build_graph()` with three nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

A conditional edge sends `policy_question` to the retrieval node and `general_question` to the direct-answer node.

## Intent Classification

The mock classifier checks for policy-related keywords:

- `delivery`
- `return`
- `refund`
- `membership`
- `tracking`
- `cancel`
- `gift card`
- `support hours`

Matching questions are classified as `policy_question`.

Other questions are classified as `general_question`.

The mock classifier does not require an external LLM.

## Structured Output

The API response is validated using Pydantic.

The response contains:

- `answer`
- `sources`
- `confidence`

Example response:

    {
      "answer": "Based on the retrieved context: ...",
      "sources": ["doc_01"],
      "confidence": 1.0
    }

The confidence value is restricted between `0.0` and `1.0`.

## MOCK_LLM Mode

The application runs in deterministic mock mode by default.

    MOCK_LLM=1

Mock mode:

- Requires no API key
- Requires no paid LLM
- Works offline
- Produces deterministic responses

`MOCK_LLM=0` is reserved for optional real LLM integration.

The graded workflow uses deterministic mock mode.

## FastAPI

FastAPI exposes the support assistant through:

    POST /ask

Example request:

    {
      "query": "What is the standard delivery fee?"
    }

Run the application locally:

    python -m uvicorn main:app --reload

API:

    http://127.0.0.1:8000

Swagger documentation:

    http://127.0.0.1:8000/docs

## API Testing

### Policy Question

Tested query:

    What is the standard delivery fee?

The API successfully returned a response using retrieved policy context.

Example response:

    {
      "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials...",
      "sources": ["doc_01"],
      "confidence": 1.0
    }

### General Question

Tested query:

    What is the capital of India?

The API returned:

    {
      "answer": "I can only answer questions about Zepto policies right now.",
      "sources": [],
      "confidence": 1.0
    }

Both policy and general-question paths were tested successfully.

## Docker

The application is designed to run using Docker.

Build the image from the repository root:

    docker build -f support_assistant/Dockerfile -t zepto-support-assistant .

Run the container:

    docker run --rm -p 7860:7860 zepto-support-assistant

API:

    http://127.0.0.1:7860

Swagger documentation:

    http://127.0.0.1:7860/docs

The Docker container serves the same `POST /ask` endpoint as the local FastAPI application.

## Project Structure

    support_assistant/
    │
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    │
    ├── main.py
    ├── Dockerfile
    └── README.md

## Technologies

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Docker