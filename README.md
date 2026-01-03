# sentinel

# RAG MVP - Day 6

A minimal RAG (Retrieval-Augmented Generation) system that retrieves relevant document chunks and uses an LLM to generate answers. **Day 3 added citation enforcement and validation. Day 4 adds conflict detection. Day 5 adds a decision engine with abstention logic, risk classification, and clarification requests. Day 6 adds an offline evaluation suite with attribute-based scoring and regression comparison.**

## Features

- **Document Ingestion**: Loads documents from JSON, chunks them by paragraphs
- **Vector Search**: Uses sentence-transformers + FAISS for semantic search
- **LLM Integration**: OpenAI API with automatic fallback to stub generator
- **Structured Citations**: LLM must return JSON with answer and citations
- **Citation Validation**: Enforces citations reference actual retrieved chunks
- **Hard Blocking**: Responses with invalid citations are BLOCKED with safe fallback
- **Conflict Detection**: Detects contradictory policy statements in retrieved chunks
- **Decision Engine**: Central guardrail that prevents fluent-but-wrong answers through abstention logic
- **Risk Classification**: Classifies queries as high/medium/low risk based on keywords
- **Abstention Logic**: ABSTAINS when retrieval confidence is low, documents are stale, or insufficient evidence
- **Clarification Requests**: CLARIFY decision for ambiguous queries that need user context
- **Retrieval Quality Signals**: Confidence metrics and freshness/staleness detection
- **Debug Endpoints**: Inspect retrieval, conflicts, and decision engine independently
- **Offline Evaluation Suite**: Golden set of test cases with attribute-based scoring
- **Regression Comparison**: Compare baseline vs candidate configs with gate rules
- **REST API**: FastAPI endpoints for querying and system info

## Project Structure

```
rag_mvp/
├── app.py                 # FastAPI application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── tests/                # Unit tests
│   ├── test_retrieval_quality.py
│   ├── test_citation_validation.py
│   ├── test_conflict_detection.py
│   ├── test_decision.py
│   └── test_eval.py
├── data/
│   └── docs.json        # Seed documents
├── eval/
│   ├── __init__.py
│   ├── golden_set_v1.json  # Golden test cases
│   ├── gate_rules.json     # Gate rules for rollout blocking
│   ├── pipeline.py         # In-process pipeline for eval
│   ├── scoring.py          # Attribute scoring functions
│   ├── gate.py             # Gate evaluation logic
│   └── run_eval.py         # Eval runner CLI
├── configs/
│   ├── baseline.json       # Baseline config
│   └── candidate.json      # Candidate config
└── rag/
    ├── __init__.py
    ├── config.py        # Configuration constants
    ├── schemas.py       # Pydantic models
    ├── ingest.py        # Document loading and indexing
    ├── retrieve.py      # Retrieval logic + quality signals
    ├── llm.py          # LLM integration (structured output)
    ├── validate.py     # Citation validation
    ├── conflicts.py    # Conflict detection
    ├── risk.py         # Risk classification
    └── decision.py     # Decision engine
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
pytest tests/test_conflict_detection.py -v
pytest tests/test_decision.py -v
pytest tests/test_eval.py -v
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

**Response (Valid Citations, No Conflicts):**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "decision": "ANSWER",
  "answer": "No, SSNs must not be stored in plaintext...",
  "citations": ["security-policy-v2.1#p1"],
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
  },
  "conflicts": {
    "conflict_detected": false,
    "conflict_type": null,
    "pairs": [],
    "summary": "No conflicts detected"
  },
  "reasons": [],
  "signals": {
    "retrieval_confidence_max": 0.85,
    "retrieval_hit_count": 3,
    "freshness_violation": false
  }
}
```

**Response (Conflict Detected - ABSTAIN):**
```json
{
  "query": "Can we store SSNs in plaintext?",
  "decision": "ABSTAIN",
  "answer": "I can't answer definitively because the retrieved sources conflict. Source A (security-policy-v2.1) says: SSNs must not be stored in plaintext... While Source B (security-policy-legacy-v1.0) says: SSNs may be stored in plaintext... Please confirm the authoritative policy or escalate to an owner.",
  "citations": ["security-policy-v2.1#p0", "security-policy-legacy-v1.0#p0"],
  "retrieval": {...},
  "retrieval_quality": {...},
  "validation": {
    "citation_valid": true,
    "errors": [],
    "warnings": []
  },
  "conflicts": {
    "conflict_detected": true,
    "conflict_type": "policy",
    "pairs": [
      {
        "chunk_a": {"doc_id": "security-policy-v2.1", "chunk_id": "security-policy-v2.1#p0"},
        "chunk_b": {"doc_id": "security-policy-legacy-v1.0", "chunk_id": "security-policy-legacy-v1.0#p0"},
        "reason": "store in plaintext vs must be encrypted / not plaintext",
        "evidence_snippets": {
          "a": "...SSNs must not be stored in plaintext...",
          "b": "...SSNs may be stored in plaintext..."
        },
        "conflict_type": "policy"
      }
    ],
    "summary": "Policy conflict detected: 1 conflicting pair(s) found"
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
  },
  "conflicts": {...},
  "reasons": ["invalid_citations"],
  "signals": {...}
}
```

**Response (Low Confidence - ABSTAIN):**
```json
{
  "query": "What is our refund policy?",
  "decision": "ABSTAIN",
  "answer": "I don't have enough reliable evidence in the retrieved documents to answer this safely.",
  "citations": [],
  "retrieval": {...},
  "retrieval_quality": {
    "confidence": {
      "max": 0.45,
      "mean": 0.40,
      "hit_count": 2
    },
    ...
  },
  "validation": {...},
  "conflicts": {...},
  "reasons": ["low_retrieval_confidence"],
  "signals": {
    "retrieval_confidence_max": 0.45,
    "retrieval_hit_count": 2,
    "freshness_violation": false
  }
}
```

**Response (Stale Documents - ABSTAIN):**
```json
{
  "query": "What is our refund policy?",
  "decision": "ABSTAIN",
  "answer": "The retrieved sources appear outdated, so I can't answer confidently.",
  "citations": [],
  "retrieval": {...},
  "retrieval_quality": {
    "freshness": {
      "freshness_violation": true,
      "freshness_violation_count": 3,
      "freshness_days": 30
    },
    ...
  },
  "validation": {...},
  "conflicts": {...},
  "reasons": ["stale_documents"],
  "signals": {
    "retrieval_confidence_max": 0.85,
    "freshness_violation": true
  }
}
```

**Response (Ambiguous Query - CLARIFY):**
```json
{
  "query": "Can we downgrade without penalties?",
  "decision": "CLARIFY",
  "answer": "Could you clarify which plan type, region, or timeframe you're asking about?",
  "citations": [],
  "retrieval": {...},
  "retrieval_quality": {...},
  "validation": {...},
  "conflicts": {...},
  "reasons": ["ambiguous_query"],
  "signals": {...}
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

### GET /debug/conflicts

Debug endpoint to inspect retrieval results, quality signals, and conflict detection without LLM call.

**Query Parameters:**
- `q` (required): Query string
- `top_k` (optional, default=5): Number of chunks to retrieve
- `freshness_days` (optional, default=90): Freshness threshold in days

**Example:**
```bash
curl "http://localhost:8000/debug/conflicts?q=Can%20we%20store%20SSNs%20in%20plaintext?&top_k=5"
```

### GET /debug/decision

Debug endpoint to inspect decision engine behavior without LLM call. Returns retrieval, conflicts, risk classification, and decision result.

**Query Parameters:**
- `q` (required): Query string
- `top_k` (optional, default=5): Number of chunks to retrieve
- `freshness_days` (optional, default=90): Freshness threshold in days

**Example:**
```bash
curl "http://localhost:8000/debug/decision?q=What%20is%20our%20refund%20policy?&top_k=5"
```

**Response:**
```json
{
  "query": "What is our refund policy?",
  "risk": {
    "risk_level": "high",
    "matched_keywords": ["refund", "policy"]
  },
  "retrieval_quality": {...},
  "conflicts": {...},
  "validation": null,
  "decision_result": {
    "decision": "ABSTAIN",
    "reasons": ["low_retrieval_confidence"],
    "user_message": "I don't have enough reliable evidence in the retrieved documents to answer this safely.",
    "thresholds": {
      "conf_max": 0.70,
      "freshness_days": 30,
      "min_chunks": 2
    },
    "signals": {
      "retrieval_confidence_max": 0.55,
      "retrieval_hit_count": 3,
      "freshness_violation": false
    },
    "risk": {...}
  }
}
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

## Decision Logic

The system uses a precedence-based decision logic with the following order:

### Decision Precedence

1. **BLOCK** (highest priority): If citations are invalid → return safe fallback, clear citations
2. **ABSTAIN** (conflicts): If conflicts are detected → surface conflict, no silent arbitration
3. **ABSTAIN** (low confidence): If retrieval confidence < threshold OR hit count < minimum → insufficient evidence
4. **ABSTAIN** (stale documents): If documents violate freshness threshold AND risk is high/medium → outdated sources
5. **CLARIFY**: If query is ambiguous (missing context like plan type, region, timeframe) → ask clarifying question
6. **ANSWER** (default): If all checks pass → return LLM-generated answer with citations

### Risk-Based Thresholds

The decision engine uses different thresholds based on query risk level:

- **High Risk** (policy, legal, compliance, security, financial):
  - Confidence threshold: 0.70
  - Freshness threshold: 30 days
  - Keywords: policy, legal, compliance, security, SSN, PII, encryption, SOC2, HIPAA, refund, chargeback, pricing, limits, retention, delete, GDPR

- **Medium Risk** (operational):
  - Confidence threshold: 0.60
  - Freshness threshold: 90 days
  - Keywords: rate limit, SLA, uptime, quota

- **Low Risk** (general/educational):
  - Confidence threshold: 0.60
  - Freshness threshold: 90 days

### Decision Types

- **ANSWER**: Normal path - LLM generates answer with citations
- **ANSWER_WITH_CAVEATS**: (Future) Answer with warnings about confidence/freshness
- **CLARIFY**: Query needs clarification - returns clarifying question, no LLM call
- **ABSTAIN**: Insufficient or unsafe evidence - returns abstention message, no LLM call
- **BLOCK**: Invalid citations - returns safe fallback, no answer provided

## Conflict Detection

The system detects conflicts between retrieved chunks using heuristic-based pattern matching:

### Conflict Types

1. **Policy Conflicts**: Detects polarity mismatches (allowed vs prohibited)
   - Keywords: `allowed/permitted/may/can` vs `prohibited/must not/may not/cannot/forbidden`
   - Examples: "SSNs may be stored" vs "SSNs must not be stored"

2. **Refund Conflicts**: Detects refund policy contradictions
   - Keywords: `refundable/refunds allowed` vs `non-refundable/no refunds`

3. **Encryption Conflicts**: Detects plaintext vs encryption requirements
   - Keywords: `store in plaintext` vs `must be encrypted/not plaintext`

4. **Numeric Conflicts**: Detects conflicting numeric values with same units
   - Examples: "1000 requests per hour" vs "300 requests per hour"
   - Only flags when both chunks discuss the same topic (e.g., "rate limit", "api rate")

### Conflict Detection Rules

- Only detects conflicts between chunks from **different documents** (same-document conflicts ignored)
- Requires **topic overlap** to avoid false positives (both chunks must mention related keywords)
- Returns conflict pairs with evidence snippets for transparency

## Example Queries

### 1. Normal Answer with Valid Citations (No Conflicts)
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the data retention policy?",
    "top_k": 5
  }'
```

This should return `decision: "ANSWER"` with valid citations and `conflicts.conflict_detected: false`.

### 2. Conflict Detection - SSN Plaintext Query
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we store SSNs in plaintext?",
    "top_k": 5
  }'
```

This should return `decision: "ABSTAIN"` with `conflicts.conflict_detected: true` and a conflict message that surfaces both conflicting sources.

### 3. Conflict Detection - API Rate Limit Query
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the API rate limit?",
    "top_k": 5
  }'
```

This should detect a numeric conflict between the 2021 policy (1000 req/hr) and 2024 policy (300 req/hr), returning `decision: "ABSTAIN"` with `conflict_type: "numeric"`.

### 4. Stub Mode (No OpenAI Key)
```bash
# Without OPENAI_API_KEY set
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the data retention policy?",
    "top_k": 5
  }'
```

Stub mode will generate answers with valid citations that reference retrieved chunks.

### 5. Test Invalid Citation (using /validate endpoint)
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

### 6. Debug Retrieval
```bash
curl "http://localhost:8000/debug/retrieval?q=Are%20refunds%20allowed?&top_k=5"
```

### 7. Debug Conflicts
```bash
curl "http://localhost:8000/debug/conflicts?q=Can%20we%20store%20SSNs%20in%20plaintext?&top_k=5"
```

This returns retrieval results, quality signals, and conflict detection without generating an answer.

### 8. ABSTAIN due to Low Confidence
```bash
# Query with low retrieval confidence (use a query that doesn't match well)
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "xyzabc random query that wont match",
    "top_k": 5
  }'
```

This should return `decision: "ABSTAIN"` with `reasons: ["low_retrieval_confidence"]` or `["insufficient_retrieval_hits"]`.

### 9. CLARIFY for Ambiguous Query
```bash
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can we downgrade without penalties?",
    "top_k": 5
  }'
```

This should return `decision: "CLARIFY"` with a clarifying question about plan type, region, or timeframe.

### 10. ABSTAIN due to Staleness (High-Risk Query)
```bash
# Query with stale documents (requires docs older than 30 days for high-risk queries)
curl -X POST "http://localhost:8000/answer" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our refund policy?",
    "top_k": 5,
    "freshness_days": 30
  }'
```

If retrieved documents are older than 30 days (high-risk freshness threshold), this should return `decision: "ABSTAIN"` with `reasons: ["stale_documents"]`.

### 11. Debug Decision Engine
```bash
curl "http://localhost:8000/debug/decision?q=What%20is%20our%20refund%20policy?&top_k=5"
```

This returns risk classification, retrieval quality, conflicts, and decision result without LLM call.

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
- `DEFAULT_CONFIDENCE_THRESHOLD = 0.60`: Default confidence threshold for low/medium risk queries
- `DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK = 0.70`: Confidence threshold for high-risk queries
- `DEFAULT_FRESHNESS_DAYS_HIGH_RISK = 30`: Freshness threshold for high-risk queries
- `DEFAULT_MIN_CHUNKS = 2`: Minimum number of retrieved chunks required

These can be overridden per request via API parameters (where applicable).

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

## Day 4 Changes

- ✅ Added conflict detection module (`rag/conflicts.py`)
- ✅ Implemented heuristic-based conflict detection (policy, numeric, refund, encryption conflicts)
- ✅ Added decision logic: BLOCK (invalid citations) > ABSTAIN (conflicts) > ANSWER
- ✅ Added conflict surfacing: system explicitly states conflicts, no silent arbitration
- ✅ Added `/debug/conflicts` endpoint for conflict detection testing
- ✅ Extended `/answer` response with conflict detection results
- ✅ Added comprehensive unit tests for conflict detection
- ✅ Updated schemas to include `ConflictResult` and `ConflictPair` types

## Day 5 Changes

- ✅ Added decision engine module (`rag/decision.py`) - central guardrail for safe responses
- ✅ Added risk classification module (`rag/risk.py`) - classifies queries as high/medium/low risk
- ✅ Implemented abstention logic based on retrieval confidence thresholds (risk-adaptive)
- ✅ Added staleness-based abstention for high/medium risk queries
- ✅ Added clarification request logic for ambiguous queries (CLARIFY decision)
- ✅ Integrated decision engine into `/answer` endpoint (called before LLM to save cost)
- ✅ Added `/debug/decision` endpoint to inspect decision engine behavior
- ✅ Extended `/answer` response with `reasons` and `signals` fields
- ✅ Updated schemas with `DecisionResult`, `RiskResult`, and enhanced `AnswerResponse`
- ✅ Added comprehensive unit tests for decision engine and risk classification
- ✅ Updated configuration with decision engine thresholds

## Day 6 Changes

- ✅ Added offline evaluation suite with golden set of 25+ test cases
- ✅ Created config system supporting overrides from JSON files
- ✅ Implemented attribute-based scoring (outcome correctness, citation validity/coverage, conflict/staleness handling, refusal correctness, unsafe answer detection)
- ✅ Added regression comparison between baseline and candidate configs
- ✅ Implemented gate rules for blocking rollouts on regressions
- ✅ Created eval runner CLI (`eval/run_eval.py`) with in-process pipeline execution
- ✅ Added comprehensive unit tests for eval components
- ✅ Evaluation runs in stub mode by default for determinism

## Offline Evaluation Suite

The evaluation suite provides a golden set of test cases and attribute-based scoring to gate rollouts and prevent regressions.

### Running Evaluation

**Basic usage (single config):**
```bash
python -m eval.run_eval \
  --suite eval/golden_set_v1.json \
  --config configs/baseline.json \
  --out eval/results.json
```

**With gate rules (blocks rollout on failure):**
```bash
python -m eval.run_eval \
  --suite eval/golden_set_v1.json \
  --config configs/candidate.json \
  --baseline eval/results_baseline.json \
  --gate eval/gate_rules.json \
  --out eval/results_candidate.json
```

**Comparing baseline vs candidate:**
```bash
# Run baseline first
python -m eval.run_eval --suite eval/golden_set_v1.json --config configs/baseline.json --out eval/results_baseline.json

# Run candidate and compare
python -m eval.run_eval \
  --suite eval/golden_set_v1.json \
  --config configs/candidate.json \
  --baseline eval/results_baseline.json \
  --gate eval/gate_rules.json \
  --out eval/results_candidate.json
```

### Evaluation Metrics

The eval suite computes the following metrics:

- **overall_pass_rate**: Percentage of test cases that pass all attribute checks
- **false_accept_rate**: Percentage of cases that answered when expected to ABSTAIN/BLOCK/CLARIFY
- **false_refuse_rate**: Percentage of cases that ABSTAINed/BLOCKed/CLARIFYed when expected to ANSWER
- **citation_validity_rate**: Percentage of cases with valid citations (when ANSWERing)
- **staleness_correct_rate**: Percentage of staleness cases handled correctly
- **conflict_correct_rate**: Percentage of conflict cases handled correctly
- **slice_metrics**: Per-slice metrics (high_risk, conflict, stale)

### Attribute Scoring

Each test case is scored across multiple attributes:

1. **outcome_correctness**: Decision matches expected outcome (with acceptable alternates)
2. **citation_validity**: Citations are valid (when ANSWERing)
3. **citation_coverage**: Required citation doc_ids are present (when specified)
4. **conflict_handling**: Conflicts are detected and surfaced (when required)
5. **staleness_handling**: Staleness is detected and handled (when required)
6. **refusal_correctness**: Refusals use safe fallback messages
7. **unsafe_answer**: Severity-0 gate - detects unsafe answers (low confidence, stale, or conflicted)

### Gate Rules

Gate rules (in `eval/gate_rules.json`) block rollouts when:

- `max_false_accept_rate`: False accept rate exceeds threshold (default: 0.0)
- `min_overall_pass_rate`: Overall pass rate below threshold (default: 0.85)
- `min_citation_validity_rate`: Citation validity below threshold (default: 0.99)
- `no_regression_slices`: Specific slices (e.g., high_risk, conflict) must not regress vs baseline

If any gate rule is violated, the eval runner exits with code 1 (blocks rollout).

### Configuration

Config files (JSON) control:
- `top_k`: Number of chunks to retrieve
- `freshness_days`: Freshness threshold (default/medium-risk)
- `freshness_days_high_risk`: Freshness threshold for high-risk queries
- `confidence_threshold`: Confidence threshold (default/medium-risk)
- `confidence_threshold_high_risk`: Confidence threshold for high-risk queries
- `min_chunks`: Minimum number of chunks required

Example config:
```json
{
  "top_k": 5,
  "freshness_days": 90,
  "freshness_days_high_risk": 30,
  "confidence_threshold": 0.60,
  "confidence_threshold_high_risk": 0.70,
  "min_chunks": 2
}
```

### Golden Set Format

Each test case in `eval/golden_set_v1.json` includes:

- `id`: Unique identifier
- `query`: Test query
- `expected_outcome`: Expected decision (ANSWER, ABSTAIN, CLARIFY, BLOCK, ANSWER_WITH_CAVEATS)
- `risk_level`: Risk level (high, medium, low)
- `required_behavior`: Object with booleans (e.g., `must_surface_conflict`, `must_detect_staleness`)
- `required_citation_doc_ids`: Optional array of doc_ids that must be cited
- `notes`: Optional description

### Important Notes

- **Evaluation is a GATE, not a report**: The primary purpose is to block rollouts when behavior regresses
- **Stub mode by default**: Evaluations run in stub LLM mode for determinism (unless `--no-stub` is provided)
- **In-process execution**: The eval runner imports pipeline functions directly (no HTTP calls)
- **Acceptable alternates**: Some outcome mismatches are acceptable (e.g., ABSTAIN for ANSWER_WITH_CAVEATS in high-risk cases)

## Next Steps (Future Days)

- Add LLM-judge scoring for answer quality (Day 6)
- Add source citation formatting in answer text
- Implement reranking based on freshness and confidence
- Add query expansion and refinement
- Implement conversation history
- Add evaluation metrics
- Human routing workflow for conflicts
- Canary rollout and config promotion gates (Day 7)

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
