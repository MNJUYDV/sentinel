"""
FastAPI application for RAG MVP Day 7.
"""
import logging
import os
from pathlib import Path

import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

from rag.config import DEFAULT_TOP_K, DEFAULT_FRESHNESS_DAYS, Config
from rag.conflicts import detect_conflicts
from rag.decision import decide
from rag.ingest import initialize_rag_index
from rag.llm import generate_answer
from rag.pipeline import run_query
from rag.retrieve import compute_retrieval_quality, retrieve
from rag.validate import validate_citations
from rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    ChunkInfo,
    ConflictPair,
    ConflictResult,
    DebugConflictsResponse,
    DebugDecisionResponse,
    DebugRetrievalResponse,
    DecisionResult,
    DocsResponse,
    DocumentInfo,
    RetrievalResult,
    RiskResult,
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
)
from registry.store import (
    list_versions, get_version, create_version, get_pointers, get_prod_history
)
from registry.promotion import promote_to_prod, rollback_prod
from registry.schemas import (
    CreateConfigRequest, PromoteRequest, RollbackRequest,
    PromoteResponse, RollbackResponse
)
from registry.models import ConfigVersion, ConfigVersionSummary, Pointers
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG MVP - Day 7",
    description="A minimal RAG system with naive retrieval, LLM answering, retrieval quality signals, citation enforcement, conflict detection, decision engine, and config versioning",
    version="0.7.0"
)

# Global state: initialized on startup
vector_index = None
documents = []

# Safe fallback answer when citations are invalid
SAFE_FALLBACK_ANSWER = "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents."


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom handler for validation errors to provide clearer messages."""
    errors = exc.errors()
    error_messages = []
    hint = None
    
    for error in errors:
        field = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        error_type = error.get("type", "")
        
        # Handle JSON decode errors
        if "json decode" in msg.lower() or error_type in ["json_invalid", "value_error.jsondecode"]:
            # Try to get the actual error position
            ctx = error.get("ctx", {})
            pos = ctx.get("pos", "unknown")
            loc = error.get("loc", [])
            
            error_messages.append(
                f"Invalid JSON in request body: {msg}. "
                f"Error location: {loc}. "
                f"Please check your JSON syntax around that position."
            )
            hint = (
                "Common JSON errors:\n"
                "- Missing commas between fields (check before position 1527)\n"
                "- Unclosed brackets [] or braces {}\n"
                "- Invalid quotes (use double quotes \" for strings)\n"
                "- Trailing commas before closing brackets\n"
                "- Special characters in text fields that need escaping\n"
                "Validate your JSON at: https://jsonlint.com/ or check the Swagger UI example"
            )
        
        # Handle freshness_days errors
        elif "freshness_days" in field:
            error_messages.append(
                f"freshness_days must be >= 0 if provided (0 to disable, >= 1 for threshold), or omit to use default ({DEFAULT_FRESHNESS_DAYS} days). Got: {error.get('input', 'unknown')}"
            )
            if hint is None:
                hint = "For freshness_days: omit the parameter to use default, or provide a value >= 0"
        
        # Handle missing required fields
        elif error_type == "missing":
            error_messages.append(f"Missing required field: {field}. This field is required.")
        
        # Generic error
        else:
            error_messages.append(f"{field}: {msg}")
    
    response_content = {"detail": error_messages}
    if hint:
        response_content["hint"] = hint
    
    return JSONResponse(
        status_code=422,
        content=response_content
    )


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG index on startup."""
    global vector_index, documents
    
    try:
        logger.info("Initializing RAG index...")
        # Get path to docs.json
        script_dir = Path(__file__).parent
        docs_path = script_dir / "data" / "docs.json"
        
        vector_index, documents = initialize_rag_index(str(docs_path))
        logger.info(f"RAG system initialized with {len(documents)} documents")
    except Exception as e:
        logger.error(f"Failed to initialize RAG index: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-mvp"}


@app.get("/docs", response_model=DocsResponse)
async def get_docs():
    """Get information about loaded documents."""
    if not documents:
        raise HTTPException(status_code=503, detail="Documents not loaded")
    
    doc_infos = [
        DocumentInfo(
            doc_id=doc["doc_id"],
            title=doc["title"],
            timestamp=doc["timestamp"]
        )
        for doc in documents
    ]
    
    return DocsResponse(count=len(documents), documents=doc_infos)


@app.post("/answer", response_model=AnswerResponse)
async def answer_query(
    request: AnswerRequest,
    env: str = Query("prod", description="Environment: dev, staging, or prod")
):
    """
    Answer a query using RAG with decision engine, citation enforcement, and conflict detection.
    Uses the active config for the specified environment (default: prod).
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Get active config for environment
        from registry.store import get_pointer, get_version
        config_id = get_pointer(env)
        if not config_id:
            raise HTTPException(status_code=500, detail=f"No config pointer for environment: {env}")
        
        config_version = get_version(config_id)
        if not config_version:
            raise HTTPException(status_code=500, detail=f"Config {config_id} not found")
        
        # Create Config object from version
        config = Config(overrides=config_version.config)
        
        # Use pipeline to run query
        mode = "stub" if not os.getenv("OPENAI_API_KEY") else "openai"
        response = run_query(request.query, vector_index, config, mode=mode)
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/validate", response_model=ValidateResponse)
async def validate_citations_endpoint(request: ValidateRequest):
    """
    Validate citations against retrieved chunks.
    Useful for testing citation validation logic independently.
    """
    try:
        # Convert ChunkInfo to dict format expected by validator
        retrieved_chunks_dict = [
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "timestamp": chunk.timestamp,
                "similarity": chunk.similarity,
                "text": chunk.text
            }
            for chunk in request.retrieved_chunks
        ]
        
        # Validate citations
        citation_valid, errors, warnings = validate_citations(
            request.citations,
            retrieved_chunks_dict,
            request.answer
        )
        
        return ValidateResponse(
            citation_valid=citation_valid,
            errors=errors,
            warnings=warnings
        )
    
    except Exception as e:
        logger.error(f"Error in validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in validation: {str(e)}")


@app.get("/debug/retrieval", response_model=DebugRetrievalResponse)
async def debug_retrieval(
    q: str = Query(..., description="Query string"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve"),
    freshness_days: int = Query(None, ge=0, description="Freshness threshold in days (optional: >= 1 for threshold, 0 to disable, None for default)")
):
    """
    Debug endpoint to inspect retrieval results and quality signals without LLM call.
    Useful for quickly inspecting retrieval behavior.
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use provided freshness_days or default
        # If freshness_days is 0, disable freshness checking
        disable_freshness = (freshness_days == 0)
        if freshness_days == 0:
            freshness_threshold = None
        elif freshness_days is not None:
            freshness_threshold = freshness_days
        else:
            freshness_threshold = DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, q, top_k)
        
        # Compute retrieval quality signals
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_threshold, disable_freshness=disable_freshness)
        
        # Format response
        chunk_infos = [
            ChunkInfo(
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                timestamp=chunk["timestamp"],
                similarity=chunk["similarity"],
                text=chunk["text"]
            )
            for chunk in retrieved_chunks
        ]
        
        response = DebugRetrievalResponse(
            query=q,
            retrieval=RetrievalResult(
                top_k=top_k,
                chunks=chunk_infos
            ),
            retrieval_quality=retrieval_quality
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in debug retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in debug retrieval: {str(e)}")


@app.get("/debug/conflicts", response_model=DebugConflictsResponse)
async def debug_conflicts(
    q: str = Query(..., description="Query string"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve"),
    freshness_days: int = Query(None, ge=0, description="Freshness threshold in days (optional: >= 1 for threshold, 0 to disable, None for default)")
):
    """
    Debug endpoint to inspect retrieval results, quality signals, and conflict detection without LLM call.
    Useful for quickly inspecting conflict detection behavior.
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use provided freshness_days or default
        # If freshness_days is 0, disable freshness checking
        disable_freshness = (freshness_days == 0)
        if freshness_days == 0:
            freshness_threshold = None
        elif freshness_days is not None:
            freshness_threshold = freshness_days
        else:
            freshness_threshold = DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, q, top_k)
        
        # Compute retrieval quality signals
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_threshold, disable_freshness=disable_freshness)
        
        # Detect conflicts
        conflict_result_dict = detect_conflicts(retrieved_chunks)
        # Convert pairs to ConflictPair objects
        conflict_pairs = [
            ConflictPair(**pair) for pair in conflict_result_dict.get("pairs", [])
        ]
        conflict_result = ConflictResult(
            conflict_detected=conflict_result_dict["conflict_detected"],
            conflict_type=conflict_result_dict.get("conflict_type"),
            pairs=conflict_pairs,
            summary=conflict_result_dict["summary"]
        )
        
        # Format response
        chunk_infos = [
            ChunkInfo(
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                timestamp=chunk["timestamp"],
                similarity=chunk["similarity"],
                text=chunk["text"]
            )
            for chunk in retrieved_chunks
        ]
        
        response = DebugConflictsResponse(
            query=q,
            retrieval=RetrievalResult(
                top_k=top_k,
                chunks=chunk_infos
            ),
            retrieval_quality=retrieval_quality,
            conflicts=conflict_result
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in debug conflicts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in debug conflicts: {str(e)}")


@app.get("/debug/decision", response_model=DebugDecisionResponse)
async def debug_decision(
    q: str = Query(..., description="Query string"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=20, description="Number of chunks to retrieve"),
    freshness_days: int = Query(None, ge=0, description="Freshness threshold in days (optional: >= 1 for threshold, 0 to disable, None for default)")
):
    """
    Debug endpoint to inspect decision engine behavior without LLM call.
    Returns retrieval, conflicts, risk classification, and decision result.
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use provided freshness_days or default
        disable_freshness = (freshness_days == 0)
        if freshness_days == 0:
            freshness_threshold = None
        elif freshness_days is not None:
            freshness_threshold = freshness_days
        else:
            freshness_threshold = DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, q, top_k)
        
        # Compute retrieval quality signals
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_threshold, disable_freshness=disable_freshness)
        
        # Detect conflicts
        conflict_result_dict = detect_conflicts(retrieved_chunks)
        conflict_pairs = [
            ConflictPair(**pair) for pair in conflict_result_dict.get("pairs", [])
        ]
        conflict_result = ConflictResult(
            conflict_detected=conflict_result_dict["conflict_detected"],
            conflict_type=conflict_result_dict.get("conflict_type"),
            pairs=conflict_pairs,
            summary=conflict_result_dict["summary"]
        )
        
        # Run decision engine (with placeholder validation since we don't have LLM answer)
        placeholder_validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        decision_result_dict = decide(
            query=q,
            retrieved_chunks=retrieved_chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflict_result,
            validation=placeholder_validation
        )
        
        # Convert decision result dict to DecisionResult schema
        risk_result_dict = decision_result_dict.get("risk", {})
        risk_result = RiskResult(
            risk_level=risk_result_dict.get("risk_level", "low"),
            matched_keywords=risk_result_dict.get("matched_keywords", [])
        )
        
        decision_result = DecisionResult(
            decision=decision_result_dict["decision"],
            reasons=decision_result_dict.get("reasons", []),
            user_message=decision_result_dict.get("user_message"),
            thresholds=decision_result_dict.get("thresholds", {}),
            signals=decision_result_dict.get("signals", {}),
            risk=risk_result
        )
        
        response = DebugDecisionResponse(
            query=q,
            risk=risk_result,
            retrieval_quality=retrieval_quality,
            conflicts=conflict_result,
            validation=None,  # Not applicable without LLM call
            decision_result=decision_result
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in debug decision: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in debug decision: {str(e)}")


# ============================================================================
# Config Registry Endpoints
# ============================================================================

@app.get("/configs", response_model=List[ConfigVersionSummary])
async def list_configs():
    """List all config versions (summary only)."""
    try:
        versions = list_versions()
        return [ConfigVersionSummary(**v.dict()) for v in versions]
    except Exception as e:
        logger.error(f"Error listing configs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error listing configs: {str(e)}")


@app.get("/configs/{config_id}", response_model=ConfigVersion)
async def get_config(config_id: str):
    """Get full config version by ID."""
    try:
        version = get_version(config_id)
        if not version:
            raise HTTPException(status_code=404, detail=f"Config {config_id} not found")
        return version
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting config {config_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting config: {str(e)}")


@app.get("/pointers", response_model=Pointers)
async def get_pointers_endpoint():
    """Get current config pointers for all environments."""
    try:
        return get_pointers()
    except Exception as e:
        logger.error(f"Error getting pointers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting pointers: {str(e)}")


@app.post("/configs", response_model=ConfigVersionSummary)
async def create_config(request: CreateConfigRequest):
    """Create a new config version."""
    try:
        version = create_version(
            parent_id=request.parent_id,
            author=request.author,
            change_reason=request.change_reason,
            config=request.config,
            prompt=request.prompt
        )
        return ConfigVersionSummary(**version.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error creating config: {str(e)}")


@app.post("/promote", response_model=PromoteResponse)
async def promote(request: PromoteRequest):
    """Promote candidate config to prod if gate passes."""
    try:
        result = promote_to_prod(request.candidate_id, request.actor)
        return PromoteResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error promoting config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error promoting config: {str(e)}")


@app.post("/rollback", response_model=RollbackResponse)
async def rollback(request: RollbackRequest):
    """Rollback prod to previous version."""
    try:
        result = rollback_prod(request.actor, request.reason)
        return RollbackResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error rolling back: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error rolling back: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
