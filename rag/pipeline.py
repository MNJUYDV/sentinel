"""
Pipeline entrypoint for running RAG queries in-process.
Used by evaluation suite and can be called directly (no HTTP).
"""
import os
from typing import Literal

from rag.config import Config
from rag.conflicts import detect_conflicts
from rag.decision import decide
from rag.ingest import VectorIndex
from rag.llm import generate_stub_answer
from rag.retrieve import compute_retrieval_quality, retrieve
from rag.schemas import (
    AnswerResponse,
    ChunkInfo,
    ConflictPair,
    ConflictResult,
    RetrievalResult,
    ValidationResult,
)
from rag.validate import validate_citations


SAFE_FALLBACK_ANSWER = "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents."


def run_query(
    query: str,
    vector_index: VectorIndex,
    config: Config,
    mode: Literal["stub", "openai"] = "stub"
) -> AnswerResponse:
    """
    Run the end-to-end RAG pipeline: retrieve -> conflicts -> decision -> (if ANSWER then LLM) -> validate -> response.
    
    Args:
        query: User query
        vector_index: VectorIndex instance (must be initialized)
        config: Config object with threshold settings
        mode: "stub" for stub LLM (deterministic, no OpenAI), "openai" for real OpenAI API
    
    Returns:
        AnswerResponse with decision, answer, citations, retrieval_quality, conflicts, validation, reasons, signals
    """
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(vector_index, query, config.top_k)
    
    # Compute retrieval quality signals
    retrieval_quality = compute_retrieval_quality(
        retrieved_chunks, config.freshness_days, disable_freshness=False
    )
    
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
    
    # Run decision engine BEFORE LLM call (use placeholder validation)
    placeholder_validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
    decision_result_dict = decide(
        query=query,
        retrieved_chunks=retrieved_chunks,
        retrieval_quality=retrieval_quality,
        conflicts=conflict_result,
        validation=placeholder_validation,
        config=config
    )
    
    decision = decision_result_dict["decision"]
    answer = decision_result_dict.get("user_message")
    citations = []
    reasons = decision_result_dict.get("reasons", [])
    signals = decision_result_dict.get("signals", {})
    
    # If decision is ANSWER or ANSWER_WITH_CAVEATS, call LLM and validate
    validation_result = placeholder_validation
    if decision in ("ANSWER", "ANSWER_WITH_CAVEATS"):
        # Generate answer - use stub if mode is stub or if no OpenAI key
        if mode == "stub" or not os.getenv("OPENAI_API_KEY"):
            llm_result = generate_stub_answer(query, retrieved_chunks)
        else:
            from rag.llm import generate_answer
            llm_result = generate_answer(query, retrieved_chunks)
        
        answer = llm_result["answer"]
        citations = llm_result["citations"]
        
        # Validate citations
        citation_valid, errors, warnings = validate_citations(
            citations, retrieved_chunks, answer
        )
        validation_result = ValidationResult(
            citation_valid=citation_valid,
            errors=errors,
            warnings=warnings
        )
        
        # If validation fails, override decision to BLOCK
        if not citation_valid:
            decision = "BLOCK"
            answer = SAFE_FALLBACK_ANSWER
            citations = []
            reasons = ["invalid_citations"] + errors[:1]
    elif decision == "ABSTAIN" and conflict_result.conflict_detected:
        # For conflict-based abstention, include conflict chunk citations
        if conflict_result.pairs:
            first_pair = conflict_result.pairs[0]
            chunk_a_id = first_pair.chunk_a["chunk_id"]
            chunk_b_id = first_pair.chunk_b["chunk_id"]
            citations = [chunk_a_id, chunk_b_id]
            citations = [c for c in citations if c in [ch["chunk_id"] for ch in retrieved_chunks]]
    
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
        query=query,
        decision=decision,
        answer=answer or "",
        citations=citations,
        retrieval=RetrievalResult(
            top_k=config.top_k,
            chunks=chunk_infos
        ),
        retrieval_quality=retrieval_quality,
        validation=validation_result,
        conflicts=conflict_result,
        reasons=reasons,
        signals=signals
    )
    
    return response

