"""
Pipeline function for running RAG system in-process (for eval).
"""
import os
from typing import Dict, Optional

from rag.config import Config
from rag.conflicts import detect_conflicts
from rag.decision import decide
from rag.llm import generate_answer, generate_stub_answer
from rag.retrieve import compute_retrieval_quality, retrieve
from rag.schemas import (
    ConflictPair,
    ConflictResult,
    ValidationResult,
)
from rag.validate import validate_citations


SAFE_FALLBACK_ANSWER = "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents."


def run_pipeline(
    query: str,
    vector_index,
    config: Config,
    use_stub_llm: bool = True,
    freshness_days_override: Optional[int] = None
) -> Dict:
    """
    Run the RAG pipeline in-process for evaluation.
    
    Args:
        query: User query
        vector_index: VectorIndex instance (must be initialized)
        config: Config object with threshold settings
        use_stub_llm: If True, force stub mode (for deterministic eval)
        freshness_days_override: Optional override for freshness_days (if None, uses config)
    
    Returns:
        Dict with keys: decision, answer, citations, retrieval_quality, conflicts, validation, reasons, signals
    """
    # Determine freshness days
    if freshness_days_override is not None:
        freshness_days = freshness_days_override
        disable_freshness = (freshness_days_override == 0)
    else:
        freshness_days = config.freshness_days
        disable_freshness = False
    
    # Retrieve relevant chunks
    retrieved_chunks = retrieve(vector_index, query, config.top_k)
    
    # Compute retrieval quality signals
    retrieval_quality = compute_retrieval_quality(
        retrieved_chunks, freshness_days, disable_freshness=disable_freshness
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
    
    # If decision is ANSWER, call LLM and validate
    validation_result = placeholder_validation
    if decision == "ANSWER":
        # Generate answer - use stub if requested or if no OpenAI key
        if use_stub_llm or not os.getenv("OPENAI_API_KEY"):
            llm_result = generate_stub_answer(query, retrieved_chunks)
        else:
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
    
    return {
        "decision": decision,
        "answer": answer or "",
        "citations": citations,
        "retrieval_quality": retrieval_quality,
        "conflicts": conflict_result,
        "validation": validation_result,
        "reasons": reasons,
        "signals": signals
    }

