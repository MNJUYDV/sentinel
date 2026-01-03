# sentinel

# RAG MVP - Day 3

A minimal RAG (Retrieval-Augmented Generation) system that retrieves relevant document chunks and uses an LLM to generate answers. **Day 3 adds citation enforcement and validation to prevent hallucinated citations.**

## Features

- **Document Ingestion**: Loads documents from JSON, chunks them by paragraphs
- **Vector Search**: Uses sentence-transformers + FAISS for semantic search
- **LLM Integration**: OpenAI API with automatic fallback to stub generator
- **Structured Citations**: LLM must return JSON with answer and citations
- **Citation Validation**: Enforces citations reference actual retrieved chunks
- **Hard Blocking**: Responses with invalid citations are BLOCKED with safe fallback
- **Retrieval Quality Signals**: Confidence metrics and freshness/staleness detection
- **Debug Endpoints**: Inspect retrieval and validate citations independently
- **REST API**: FastAPI endpoints for querying and system info

## Project Structure

```
rag_mvp/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── tests/                # Unit tests
│   ├── test_retrieval_quality.py
│   └── test_citation_validation.py
├── data/
│   └── docs.json        # Seed documents
└── rag/
    ├── __init__.py
    ├── config.py        # Configuration constants
    ├── schemas.py       # Pydantic models
    ├── ingest.py        # Document loading and indexing
    ├── retrieve.py      # Retrieval logic + quality signals
    ├── llm.py          # LLM integration (structured output)
    └── validate.py     # Citation validation
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

If `OPENAI_API_KEY` is not set, the system will use a stub answer generator that still demonstrates the RAG pipeline with valid citations.

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

Or run specific test suites:
```bash
pytest tests/test_retrieval_quality.py -v
pytest tests/test_citation_validation.py -v
```

## API Endpoints

### POST /answer

Answer a query using RAG retrieval and generation with citation enforcement.

**Request:**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "top_k": 5,
  "freshness_days": 90
}
```

**Response (Valid Citations):**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "decision": "ANSWER",
  "answer": "No, SSNs must not be stored in plaintext...",
  "citations": ["security-policy-v2.1#p1", "security-policy-legacy-v1.0#p1"],
  "retrieval": {
    "top_k": 5,
    "chunks": [...]
  },
  "retrieval_quality": {
    "confidence": {...},
    "freshness": {...},
    "top_doc_ids": [...],
    "top_timestamps": [...]
  },
  "validation": {
    "citation_valid": true,
    "errors": [],
    "warnings": []
  }
}
```

**Response (Invalid Citations - BLOCKED):**
```json
{
  "query": "...",
  "decision": "BLOCK",
  "answer": "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents.",
  "citations": [],
  "retrieval": {...},
  "retrieval_quality": {...},
  "validation": {
    "citation_valid": false,
    "errors": [
      "Citation 'invalid-chunk-id' does not match any retrieved chunk"
    ],
    "warnings": []
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we store SSNs in plaintext?",
    "top_k": 5
  }'
```

### POST /validate

Validate citations against retrieved chunks independently.

**Request:**
```json
{
  "answer": "SSNs must be encrypted at rest.",
  "citations": ["security-policy-v2.1#p1"],
  "retrieved_chunks": [
    {
      "doc_id": "security-policy-v2.1",
      "chunk_id": "security-policy-v2.1#p1",
      "timestamp": "2024-03-15T10:00:00Z",
      "similarity": 0.9,
      "text": "SSNs must not be stored in plaintext; encrypt at rest."
    }
  ]
}
```

**Response:**
```json
{
  "citation_valid": true,
  "errors": [],
  "warnings": []
}
```

**Example with invalid citation:**
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "Some answer text",
    "citations": ["invalid-chunk-id#p0"],
    "retrieved_chunks": [
      {
        "doc_id": "doc1",
        "chunk_id": "doc1#p0",
        "timestamp": "2024-01-01T00:00:00Z",
        "similarity": 0.9,
        "text": "Content"
      }
    ]
  }'
```

### GET /debug/retrieval

Debug endpoint to inspect retrieval results and quality signals without LLM call.

**Query Parameters:**
- `q` (required): Query string
- `top_k` (optional, default=5): Number of chunks to retrieve
- `freshness_days` (optional, default=90): Freshness threshold in days

**Example:**
```bash
curl "http://localhost:8000/debug/retrieval?q=Can%20we%20store%20SSNs%20in%20plaintext?&top_k=5"
```

### GET /health

Health check endpoint.

### GET /docs

Get information about loaded documents.

## Citation Validation Rules

The system enforces the following validation rules:

1. **Citation IDs must exist**: Every citation must exactly match a retrieved chunk ID (format: `doc_id#chunk_id`)
2. **No duplicates**: Citations must be unique
3. **Maximum citations**: Limited to 5 citations per answer
4. **Empty citations**: Only allowed when answer indicates insufficient context
5. **Keyword overlap warning**: If cited chunks share no keywords with the answer (non-blocking warning)

**Validation Errors** (cause BLOCK):
- Citation ID not in retrieved chunks
- Duplicate citations
- More than 5 citations
- Empty citations with non-insufficient answer

**Validation Warnings** (non-blocking):
- No keyword overlap between answer and cited chunks

## Retrieval Quality Signals

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

## Example Queries

### 1. Normal Answer with Valid Citations
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we store SSNs in plaintext?",
    "top_k": 5
  }'
```

This should return `decision: "ANSWER"` with valid citations.

### 2. Stub Mode (No OpenAI Key)
```bash
# Without OPENAI_API_KEY set
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the API rate limit?",
    "top_k": 5
  }'
```

Stub mode will generate answers with valid citations that reference retrieved chunks.

### 3. Test Invalid Citation (using /validate endpoint)
```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "This is a test answer",
    "citations": ["fake-doc#p0"],
    "retrieved_chunks": [
      {
        "doc_id": "real-doc",
        "chunk_id": "real-doc#p0",
        "timestamp": "2024-01-01T00:00:00Z",
        "similarity": 0.9,
        "text": "Real content"
      }
    ]
  }'
```

This demonstrates how invalid citations are detected.

### 4. Debug Retrieval
```bash
curl "http://localhost:8000/debug/retrieval?q=Are%20refunds%20allowed?&top_k=5"
```

## Seed Documents

The system comes with 10 seed documents in `data/docs.json` that include:
- **Conflicting policies**: Current security policy vs legacy policy on SSN storage
- **Stale information**: Old API rate limits (2021) vs current ones (2024)
- **Exception cases**: General refund policy vs promotional purchase exceptions
- **Retention policies**: Standard retention vs legal hold exceptions

These are designed to test how the RAG system handles conflicts, stale data, and exceptions.

## Configuration

Default configuration is in `rag/config.py`:
- `DEFAULT_TOP_K = 5`: Default number of chunks to retrieve
- `DEFAULT_FRESHNESS_DAYS = 90`: Default freshness threshold in days

These can be overridden per request via API parameters.

## Implementation Notes

- **Chunking**: Simple paragraph-based chunking (split on blank lines)
- **Embeddings**: Uses `all-MiniLM-L6-v2` from sentence-transformers
- **Vector Search**: FAISS with L2 distance on normalized embeddings
- **LLM**: OpenAI GPT-3.5-turbo with JSON structured output requirement
- **Citations**: Must be in format `doc_id#chunk_id` matching retrieved chunks
- **Blocking**: Hard block on invalid citations (severity-0 guardrail)
- **Stub Mode**: Generates valid citations even without OpenAI key

## Day 3 Changes

- ✅ Changed LLM output to structured JSON with answer and citations
- ✅ Added citation validation module (`rag/validate.py`)
- ✅ Added hard blocking for invalid citations
- ✅ Added `/validate` endpoint for independent citation validation
- ✅ Updated stub mode to return structured JSON with valid citations
- ✅ Added comprehensive unit tests for citation validation
- ✅ Added keyword overlap heuristic (warning only, non-blocking)
- ✅ Extended `/answer` response with validation signals

## Next Steps (Future Days)

- Add abstention logic based on retrieval quality (Day 5)
- Implement conflict detection and resolution
- Add source citation formatting in answer text
- Implement reranking based on freshness and confidence
- Add query expansion and refinement
- Implement conversation history
- Add evaluation metrics

## Troubleshooting

**Issue**: Import errors for sentence-transformers or faiss
- **Solution**: Make sure you've installed all dependencies: `pip install -r requirements.txt`

**Issue**: OpenAI API errors
- **Solution**: The system will automatically fall back to stub mode if the API key is missing or invalid. Check your `OPENAI_API_KEY` environment variable.

**Issue**: Responses are BLOCKED unexpectedly
- **Solution**: Check the `validation.errors` field in the response. Common issues:
  - Citations don't match retrieved chunk IDs
  - Empty citations without insufficiency message
  - More than 5 citations

**Issue**: Server fails to start
- **Solution**: Check that `data/docs.json` exists and is valid JSON. Check logs for specific error messages.

**Issue**: Tests fail
- **Solution**: Make sure pytest is installed: `pip install pytest`. Run tests from project root: `pytest tests/`
