# sentinel

# RAG MVP - Day 1

A minimal RAG (Retrieval-Augmented Generation) system that retrieves relevant document chunks and uses an LLM to generate answers.

## Features

- **Document Ingestion**: Loads documents from JSON, chunks them by paragraphs
- **Vector Search**: Uses sentence-transformers + FAISS for semantic search
- **LLM Integration**: OpenAI API with automatic fallback to stub generator
- **REST API**: FastAPI endpoints for querying and system info

## Project Structure

```
rag_mvp/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/
│   └── docs.json        # Seed documents
└── rag/
    ├── __init__.py
    ├── schemas.py       # Pydantic models
    ├── ingest.py        # Document loading and indexing
    ├── retrieve.py      # Retrieval logic
    └── llm.py          # LLM integration
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. (Optional) Set OpenAI API key for real LLM responses:
   
   Option A: Create a `.env` file (recommended):
   ```bash
   cp .env.example .env
   # Then edit .env and add your API key
   ```
   
   Option B: Set environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

If `OPENAI_API_KEY` is not set, the system will use a stub answer generator that still demonstrates the RAG pipeline.

## Running the Server

```bash
uvicorn app:app --reload
```

The server will start on `http://localhost:8000`

- API docs (Swagger UI): http://localhost:8000/docs
- Alternative API docs (ReDoc): http://localhost:8000/redoc

## API Endpoints

### POST /answer

Answer a query using RAG retrieval and generation.

**Request:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "answer": "...",
  "retrieval": {
    "top_k": 5,
    "chunks": [
      {
        "doc_id": "security-policy-v2.1",
        "chunk_id": "security-policy-v2.1#p1",
        "timestamp": "2024-03-15T10:00:00Z",
        "similarity": 0.8234,
        "text": "SSNs must not be stored in plaintext; encrypt at rest."
      }
    ]
  }
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "rag-mvp"
}
```

### GET /docs

Get information about loaded documents.

**Response:**
```json
{
  "count": 10,
  "documents": [
    {
      "doc_id": "security-policy-v2.1",
      "title": "Data Security Policy - Version 2.1",
      "timestamp": "2024-03-15T10:00:00Z"
    }
  ]
}
```

## Example Queries

Try these curl commands to test the system:

### 1. Conflicting Policy Query
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we store SSNs in plaintext?",
    "top_k": 5
  }'
```

This should retrieve both the conflicting policy documents (current vs legacy).

### 2. Stale vs Current Information
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current API rate limit?",
    "top_k": 5
  }'
```

This should retrieve both the 2021 and 2024 rate limit docs, allowing you to see how the system handles stale information.

### 3. Exception Split Risk
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Are refunds allowed for promotional purchases?",
    "top_k": 5
  }'
```

This tests whether the system retrieves both the general refund policy and the exception about promotional purchases.

### 4. Retention Policy
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How long do we retain deleted messages?",
    "top_k": 5
  }'
```

This should retrieve the retention policy with the exception for legal hold.

## Seed Documents

The system comes with 10 seed documents in `data/docs.json` that include:
- **Conflicting policies**: Current security policy vs legacy policy on SSN storage
- **Stale information**: Old API rate limits vs current ones
- **Exception cases**: General refund policy vs promotional purchase exceptions
- **Retention policies**: Standard retention vs legal hold exceptions

These are designed to test how the naive RAG system handles conflicts, stale data, and exceptions that may be split across chunks.

## Implementation Notes

- **Chunking**: Simple paragraph-based chunking (split on blank lines). Easy to modify later.
- **Embeddings**: Uses `all-MiniLM-L6-v2` from sentence-transformers (small, fast model)
- **Vector Search**: FAISS with L2 distance on normalized embeddings (equivalent to cosine similarity)
- **LLM**: OpenAI GPT-3.5-turbo with fallback stub generator
- **No guardrails**: This is intentionally naive for Day 1. No fact-checking, conflict resolution, or source validation yet.

## Next Steps (Future Days)

- Add guardrails and conflict detection
- Implement more sophisticated chunking strategies
- Add source citation and confidence scores
- Implement reranking
- Add query expansion and refinement
- Implement conversation history
- Add evaluation metrics

## Troubleshooting

**Issue**: Import errors for sentence-transformers or faiss
- **Solution**: Make sure you've installed all dependencies: `pip install -r requirements.txt`

**Issue**: OpenAI API errors
- **Solution**: The system will automatically fall back to stub mode if the API key is missing or invalid. Check your `OPENAI_API_KEY` environment variable.

**Issue**: Server fails to start
- **Solution**: Check that `data/docs.json` exists and is valid JSON. Check logs for specific error messages.

# sentinel
