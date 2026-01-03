"""
FastAPI application for RAG MVP Day 3.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

# Load environment variables from .env file
load_dotenv()

from rag.config import DEFAULT_TOP_K, DEFAULT_FRESHNESS_DAYS
from rag.ingest import initialize_rag_index
from rag.llm import generate_answer
from rag.retrieve import compute_retrieval_quality, retrieve
from rag.validate import validate_citations
from rag.schemas import (
    AnswerRequest,
    AnswerResponse,
    ChunkInfo,
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
    title="RAG MVP - Day 3",
    description="A minimal RAG system with naive retrieval, LLM answering, retrieval quality signals, and citation enforcement",
    version="0.3.0"
)

# Global state: initialized on startup
vector_index = None
documents = []

# Safe fallback answer when citations are invalid
SAFE_FALLBACK_ANSWER = "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents."


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
    Answer a query using RAG with citation enforcement:
    1. Retrieve top-k relevant chunks
    2. Generate answer with citations using LLM
    3. Validate citations against retrieved chunks
    4. BLOCK response if citations are invalid
    5. Compute and return retrieval quality signals
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use request freshness_days or default
        freshness_days = request.freshness_days if request.freshness_days is not None else DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, request.query, request.top_k)
        
        # Compute retrieval quality signals
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_days)
        
        # Generate answer with citations
        llm_result = generate_answer(request.query, retrieved_chunks)
        answer = llm_result["answer"]
        citations = llm_result["citations"]
        
        # Validate citations
        citation_valid, errors, warnings = validate_citations(
            citations, retrieved_chunks, answer
        )
        
        # Log retrieval and validation signals
        top_doc_ids_str = ", ".join(retrieval_quality.top_doc_ids[:3])
        logger.info(
            f"Query: {request.query[:50]}, "
            f"top_k: {request.top_k}, "
            f"retrieval_confidence_max: {retrieval_quality.confidence.max:.4f}, "
            f"freshness_violation: {retrieval_quality.freshness.freshness_violation}, "
            f"citation_valid: {citation_valid}, "
            f"top_doc_ids: [{top_doc_ids_str}]"
        )
        
        # Determine decision: BLOCK if citations invalid
        if not citation_valid:
            decision = "BLOCK"
            answer = SAFE_FALLBACK_ANSWER
            citations = []  # Clear invalid citations
            logger.warning(f"BLOCKED answer due to invalid citations. Errors: {errors}")
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
            )
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
    freshness_days: int = Query(None, ge=1, description="Freshness threshold in days (optional)")
):
    """
    Debug endpoint to inspect retrieval results and quality signals without LLM call.
    Useful for quickly inspecting retrieval behavior.
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Use provided freshness_days or default
        freshness_threshold = freshness_days if freshness_days is not None else DEFAULT_FRESHNESS_DAYS
        
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, q, top_k)
        
        # Compute retrieval quality signals
        retrieval_quality = compute_retrieval_quality(retrieved_chunks, freshness_threshold)
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
