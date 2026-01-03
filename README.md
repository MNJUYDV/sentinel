# sentinel

# RAG MVP - Day 2

A minimal RAG (Retrieval-Augmented Generation) system that retrieves relevant document chunks and uses an LLM to generate answers. **Day 2 adds retrieval quality signals and diagnostic endpoints.**

## Features

- **Document Ingestion**: Loads documents from JSON, chunks them by paragraphs
- **Vector Search**: Uses sentence-transformers + FAISS for semantic search
- **LLM Integration**: OpenAI API with automatic fallback to stub generator
- **Retrieval Quality Signals**: Confidence metrics and freshness/staleness detection
- **Debug Endpoint**: Inspect retrieval results without LLM calls
- **REST API**: FastAPI endpoints for querying and system info

## Project Structure

```
rag_mvp/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── tests/                # Unit tests
│   └── test_retrieval_quality.py
├── data/
│   └── docs.json        # Seed documents
└── rag/
    ├── __init__.py
    ├── config.py        # Configuration constants
    ├── schemas.py       # Pydantic models
    ├── ingest.py        # Document loading and indexing
    ├── retrieve.py      # Retrieval logic + quality signals
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
   echo "OPENAI_API_KEY=your-api-key-here" > .env
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

## Running Tests

```bash
pytest tests/
```

Or run a specific test:
```bash
pytest tests/test_retrieval_quality.py -v
```

## API Endpoints

### POST /answer

Answer a query using RAG retrieval and generation with quality signals.

**Request:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "top_k": 5,
  "freshness_days": 90
}
```

**Response:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "answer": "...",
  "retrieval": {
    "top_k": 5,
    "chunks": [...]
  },
  "retrieval_quality": {
    "confidence": {
      "max": 0.9141,
      "mean": 0.7234,
      "gap": 0.0890,
      "hit_count": 5
    },
    "freshness": {
      "oldest_timestamp": "2020-06-10T14:30:00Z",
      "newest_timestamp": "2024-03-15T10:00:00Z",
      "freshness_violation_count": 1,
      "freshness_violation": true,
      "freshness_days": 90
    },
    "top_doc_ids": ["security-policy-legacy-v1.0", "security-policy-v2.1", ...],
    "top_timestamps": ["2020-06-10T14:30:00Z", "2024-03-15T10:00:00Z", ...]
  }
}
```

### GET /debug/retrieval

Debug endpoint to inspect retrieval results and quality signals **without** LLM call. Useful for quickly inspecting retrieval behavior.

**Query Parameters:**
- `q` (required): Query string
- `top_k` (optional, default=5): Number of chunks to retrieve
- `freshness_days` (optional, default=90): Freshness threshold in days

**Example:**
```bash
curl "http://localhost:8000/debug/retrieval?q=Can%20we%20store%20SSNs%20in%20plaintext?&top_k=5&freshness_days=90"
```

**Response:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "retrieval": {
    "top_k": 5,
    "chunks": [...]
  },
  "retrieval_quality": {
    "confidence": {...},
    "freshness": {...},
    "top_doc_ids": [...],
    "top_timestamps": [...]
  }
}
```

**Using different freshness thresholds:**
```bash
# Use 30-day freshness threshold (stricter)
curl "http://localhost:8000/debug/retrieval?q=What%20is%20the%20API%20rate%20limit?&freshness_days=30"

# Use 180-day freshness threshold (more lenient)
curl "http://localhost:8000/debug/retrieval?q=What%20is%20the%20API%20rate%20limit?&freshness_days=180"
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

## Retrieval Quality Signals (Day 2)

### Confidence Signals

- **max**: Maximum similarity score among retrieved chunks (0.0-1.0)
- **mean**: Mean similarity score across all retrieved chunks
- **gap**: Difference between top1 and top2 similarity (null if only 1 chunk retrieved)
- **hit_count**: Number of chunks actually retrieved

### Freshness Signals

- **oldest_timestamp**: Oldest document timestamp among retrieved chunks
- **newest_timestamp**: Newest document timestamp among retrieved chunks
- **freshness_violation_count**: Count of chunks older than the threshold
- **freshness_violation**: Boolean indicating if any violations exist
- **freshness_days**: The freshness threshold used (default: 90 days)

These signals help identify:
- **Low confidence retrievals**: When similarity scores are low, retrieval may be poor
- **Stale information**: When retrieved documents are outdated
- **High confidence gaps**: When top result is much better than second (indicates clear winner)

## Example Queries

### 1. Conflicting Policy Query
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we store SSNs in plaintext?",
    "top_k": 5
  }'
```

This should retrieve both the conflicting policy documents (current vs legacy) and show freshness violations for the legacy document.

### 2. Stale vs Current Information (with freshness override)
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the current API rate limit?",
    "top_k": 5,
    "freshness_days": 30
  }'
```

This will use a stricter 30-day freshness threshold and should show violations for the 2021 document.

### 3. Debug Retrieval (without LLM)
```bash
curl "http://localhost:8000/debug/retrieval?q=Are%20refunds%20allowed%20for%20promotional%20purchases?&top_k=5"
```

This returns only retrieval results and quality signals, useful for debugging.

### 4. Retention Policy
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How long do we retain deleted messages?",
    "top_k": 5
  }'
```

## Seed Documents

The system comes with 10 seed documents in `data/docs.json` that include:
- **Conflicting policies**: Current security policy vs legacy policy on SSN storage
- **Stale information**: Old API rate limits (2021) vs current ones (2024)
- **Exception cases**: General refund policy vs promotional purchase exceptions
- **Retention policies**: Standard retention vs legal hold exceptions

These are designed to test how the naive RAG system handles conflicts, stale data, and exceptions that may be split across chunks.

## Configuration

Default configuration is in `rag/config.py`:
- `DEFAULT_TOP_K = 5`: Default number of chunks to retrieve
- `DEFAULT_FRESHNESS_DAYS = 90`: Default freshness threshold in days

These can be overridden per request via API parameters.

## Implementation Notes

- **Chunking**: Simple paragraph-based chunking (split on blank lines). Easy to modify later.
- **Embeddings**: Uses `all-MiniLM-L6-v2` from sentence-transformers (small, fast model)
- **Vector Search**: FAISS with L2 distance on normalized embeddings (equivalent to cosine similarity)
- **LLM**: OpenAI GPT-3.5-turbo with fallback stub generator
- **Quality Signals**: Computed on every retrieval, logged for analysis
- **No guardrails yet**: Signals are computed but don't block answers (Day 3+ feature)

## Day 2 Changes

- ✅ Added retrieval quality signals (confidence and freshness)
- ✅ Added `/debug/retrieval` endpoint for inspection
- ✅ Added configurable freshness thresholds
- ✅ Added logging of retrieval signals
- ✅ Added unit tests for quality signal computation
- ✅ Extended `/answer` endpoint to return quality signals

## Next Steps (Future Days)

- Add guardrails that use quality signals to abstain from answering
- Implement conflict detection and resolution
- Add source citation and confidence scores to answers
- Implement reranking based on freshness and confidence
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

**Issue**: Tests fail
- **Solution**: Make sure pytest is installed: `pip install pytest`. Run tests from project root: `pytest tests/`
