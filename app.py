"""
FastAPI application for RAG MVP Day 3.
"""
import logging
from pathlib import Path

import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

from rag.config import DEFAULT_TOP_K, DEFAULT_FRESHNESS_DAYS
from rag.conflicts import detect_conflicts
from rag.ingest import initialize_rag_index
from rag.llm import generate_answer
from rag.retrieve import compute_retrieval_quality, retrieve
from rag.validate import validate_citations
from rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    ChunkInfo,
    ConflictPair,
    ConflictResult,
    DebugConflictsResponse,
    DebugRetrievalResponse,
    DocsResponse,
    DocumentInfo,
    RetrievalResult,
    ValidateRequest,
    ValidateResponse,
    ValidationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG MVP - Day 4",
    description="A minimal RAG system with naive retrieval, LLM answering, retrieval quality signals, citation enforcement, and conflict detection",
    version="0.4.0"
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
async def answer_query(request: AnswerRequest):
    """
    Answer a query using RAG with citation enforcement and conflict detection:
    1. Retrieve top-k relevant chunks
    2. Detect conflicts in retrieved chunks
    3. Generate answer with citations using LLM
    4. Validate citations against retrieved chunks
    5. Decision logic: BLOCK (invalid citations) > ABSTAIN (conflicts) > ANSWER
    6. Compute and return retrieval quality signals
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use request freshness_days or default
        # If freshness_days is 0, pass None with a flag or use a special value that compute_retrieval_quality recognizes
        # We'll use None and handle 0 specially in compute_retrieval_quality
        if request.freshness_days == 0:
            freshness_days = None  # Will be handled specially in compute_retrieval_quality
        elif request.freshness_days is not None:
            freshness_days = request.freshness_days
        else:
            freshness_days = DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, request.query, request.top_k)
        
        # Compute retrieval quality signals
        # If freshness_days was 0 in request, disable freshness checking
        disable_freshness = (request.freshness_days == 0)
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_days, disable_freshness=disable_freshness)
        
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
        
        # Generate answer with citations (only if we might need it)
        llm_result = generate_answer(request.query, retrieved_chunks)
        answer = llm_result["answer"]
        citations = llm_result["citations"]
        
        # Validate citations
        citation_valid, errors, warnings = validate_citations(
            citations, retrieved_chunks, answer
        )
        
        # Log retrieval, validation, and conflict signals
        top_doc_ids_str = ", ".join(retrieval_quality.top_doc_ids[:3])
        logger.info(
            f"Query: {request.query[:50]}, "
            f"top_k: {request.top_k}, "
            f"retrieval_confidence_max: {retrieval_quality.confidence.max:.4f}, "
            f"freshness_violation: {retrieval_quality.freshness.freshness_violation}, "
            f"citation_valid: {citation_valid}, "
            f"conflict_detected: {conflict_result.conflict_detected}, "
            f"top_doc_ids: [{top_doc_ids_str}]"
        )
        
        # Decision precedence: BLOCK (citation invalid) > ABSTAIN (conflict) > ANSWER
        if not citation_valid:
            decision = "BLOCK"
            answer = SAFE_FALLBACK_ANSWER
            citations = []  # Clear invalid citations
            logger.warning(f"BLOCKED answer due to invalid citations. Errors: {errors}")
        elif conflict_result.conflict_detected and citation_valid:
            decision = "ABSTAIN"
            # Generate conflict-safe answer
            if conflict_result.pairs:
                first_pair = conflict_result.pairs[0]
                chunk_a_id = first_pair.chunk_a["chunk_id"]
                chunk_b_id = first_pair.chunk_b["chunk_id"]
                
                # Find the actual chunks for better context
                chunk_a_text = next(
                    (c["text"] for c in retrieved_chunks if c["chunk_id"] == chunk_a_id),
                    "Source A"
                )
                chunk_b_text = next(
                    (c["text"] for c in retrieved_chunks if c["chunk_id"] == chunk_b_id),
                    "Source B"
                )
                
                # Extract short snippets
                snippet_a = chunk_a_text[:150] + "..." if len(chunk_a_text) > 150 else chunk_a_text
                snippet_b = chunk_b_text[:150] + "..." if len(chunk_b_text) > 150 else chunk_b_text
                
                answer = (
                    f"I can't answer definitively because the retrieved sources conflict. "
                    f"Source A ({first_pair.chunk_a['doc_id']}) says: {snippet_a} "
                    f"While Source B ({first_pair.chunk_b['doc_id']}) says: {snippet_b} "
                    f"Please confirm the authoritative policy or escalate to an owner."
                )
                # Include both conflicting chunk IDs in citations
                citations = [chunk_a_id, chunk_b_id]
                # Ensure citations are valid (they should be since they're from retrieved chunks)
                citations = [c for c in citations if c in [ch["chunk_id"] for ch in retrieved_chunks]]
            else:
                answer = (
                    "I can't answer definitively because the retrieved sources contain conflicting information. "
                    "Please confirm the authoritative policy or escalate to an owner."
                )
            
            logger.warning(
                f"ABSTAINED due to conflict: type={conflict_result.conflict_type}, "
                f"doc_ids={[p.chunk_a['doc_id'] for p in conflict_result.pairs] + [p.chunk_b['doc_id'] for p in conflict_result.pairs]}"
            )
        else:
            decision = "ANSWER"
            if warnings:
                logger.info(f"Validation warnings: {warnings}")
        
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
        
        response = AnswerResponse(
            query=request.query,
            decision=decision,
            answer=answer,
            citations=citations,
            retrieval=RetrievalResult(
                top_k=request.top_k,
                chunks=chunk_infos
            ),
            retrieval_quality=retrieval_quality,
            validation=ValidationResult(
                citation_valid=citation_valid,
                errors=errors,
                warnings=warnings
            ),
            conflicts=conflict_result
        )
        
        return response
    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
