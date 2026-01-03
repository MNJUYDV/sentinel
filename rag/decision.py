"""
Decision engine for RAG safety system.
Determines whether to ANSWER, ABSTAIN, CLARIFY, or BLOCK based on retrieval quality,
risk level, conflicts, and validation results.
"""
import logging
from typing import Dict, List, Optional, Tuple

from rag.config import (
    Config,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK,
    DEFAULT_FRESHNESS_DAYS,
    DEFAULT_FRESHNESS_DAYS_HIGH_RISK,
    DEFAULT_MIN_CHUNKS,
    get_default_config
)
from rag.risk import classify_risk
from rag.schemas import ConflictResult, RetrievalQuality, ValidationResult

logger = logging.getLogger(__name__)


def is_ambiguous(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if query is ambiguous and needs clarification.
    
    Args:
        query: User query string
    
    Returns:
        Tuple of (is_ambiguous: bool, clarifying_question: Optional[str])
    """
    query_lower = query.lower()
    
    # Check for "can we" / "is it allowed" patterns without scope
    permission_patterns = ["can we", "can i", "is it allowed", "is allowed", "may we", "may i"]
    scope_keywords = ["plan", "subscription", "region", "timeframe", "account", "tier", "date"]
    
    has_permission_pattern = any(pattern in query_lower for pattern in permission_patterns)
    has_scope = any(keyword in query_lower for keyword in scope_keywords)
    
    if has_permission_pattern and not has_scope:
        return (True, "Could you clarify which plan type, region, or timeframe you're asking about?")
    
    # Check for "current" / "latest" without date context
    temporal_patterns = ["current", "latest", "now", "today"]
    date_keywords = ["2024", "2025", "january", "february", "march", "april", "may", "june",
                     "july", "august", "september", "october", "november", "december",
                     "q1", "q2", "q3", "q4"]
    
    has_temporal = any(pattern in query_lower for pattern in temporal_patterns)
    has_date = any(keyword in query_lower for keyword in date_keywords)
    
    if has_temporal and not has_date:
        # Check if it's asking about current state vs historical
        if "when" not in query_lower and "date" not in query_lower:
            return (True, "Could you specify the timeframe or date you're interested in?")
    
    # Check for vague action queries
    vague_actions = ["downgrade", "upgrade", "cancel", "change"]
    if any(action in query_lower for action in vague_actions):
        # Check if there's enough context about what's being changed
        context_keywords = ["plan", "subscription", "service", "tier", "feature"]
        if not any(keyword in query_lower for keyword in context_keywords):
            return (True, "Could you clarify what you'd like to change or modify?")
    
    return (False, None)


def decide(
    query: str,
    retrieved_chunks: List[dict],
    retrieval_quality: RetrievalQuality,
    conflicts: ConflictResult,
    validation: ValidationResult,
    risk_level: Optional[str] = None,
    config: Optional[Config] = None
) -> Dict:
    """
    Decide whether to ANSWER, ABSTAIN, CLARIFY, or BLOCK.
    
    Decision precedence:
    A) If validation.citation_valid == false => BLOCK
    B) Else if conflicts.conflict_detected == true => ABSTAIN (with conflict message)
    C) Else:
       - Determine risk_level
       - Check retrieval confidence thresholds
       - Check freshness violations
       - Check ambiguity
       - Otherwise => ANSWER
    
    Args:
        query: User query
        retrieved_chunks: List of retrieved chunk dictionaries
        retrieval_quality: RetrievalQuality object
        conflicts: ConflictResult object
        validation: ValidationResult object
        risk_level: Optional pre-computed risk level (if None, will compute)
    
    Returns:
        Dict with:
        - decision: "ANSWER" | "ANSWER_WITH_CAVEATS" | "CLARIFY" | "ABSTAIN" | "BLOCK"
        - reasons: List[str] - reasons for the decision
        - user_message: str - message to show user (if not ANSWER)
        - thresholds: Dict with threshold values used
        - signals: Dict with signal values (confidence, freshness, etc.)
        - risk: Dict with risk classification results
    """
    reasons = []
    signals = {}
    thresholds = {}
    
    # Use config if provided, otherwise use defaults
    if config is None:
        config = get_default_config()
    
    # Get risk classification if not provided
    if risk_level is None:
        risk_result = classify_risk(query)
        risk_level = risk_result["risk_level"]
    else:
        risk_result = classify_risk(query)
    
    # Set thresholds based on risk level using config
    confidence_threshold = config.get_confidence_threshold(risk_level)
    freshness_days = config.get_freshness_days(risk_level)
    
    thresholds = {
        "conf_max": confidence_threshold,
        "freshness_days": freshness_days,
        "min_chunks": config.min_chunks
    }
    
    signals = {
        "retrieval_confidence_max": retrieval_quality.confidence.max,
        "retrieval_hit_count": retrieval_quality.confidence.hit_count,
        "freshness_violation": retrieval_quality.freshness.freshness_violation
    }
    
    # Decision precedence: A) BLOCK (invalid citations)
    if not validation.citation_valid:
        logger.warning(f"BLOCK decision: invalid citations. Errors: {validation.errors}")
        return {
            "decision": "BLOCK",
            "reasons": ["invalid_citations"] + validation.errors[:1],  # Include first error as reason
            "user_message": "I can't provide a cited answer because the citations couldn't be verified against the retrieved documents.",
            "thresholds": thresholds,
            "signals": signals,
            "risk": risk_result
        }
    
    # Decision precedence: B) ABSTAIN (conflicts detected)
    if conflicts.conflict_detected:
        logger.warning(f"ABSTAIN decision: conflicts detected. Type: {conflicts.conflict_type}")
        # Generate conflict message (similar to current app.py logic)
        if conflicts.pairs:
            first_pair = conflicts.pairs[0]
            chunk_a_id = first_pair.chunk_a["chunk_id"]
            chunk_b_id = first_pair.chunk_b["chunk_id"]
            
            # Find chunks for snippets
            chunk_a_text = next(
                (c["text"] for c in retrieved_chunks if c["chunk_id"] == chunk_a_id),
                "Source A"
            )
            chunk_b_text = next(
                (c["text"] for c in retrieved_chunks if c["chunk_id"] == chunk_b_id),
                "Source B"
            )
            
            snippet_a = chunk_a_text[:150] + "..." if len(chunk_a_text) > 150 else chunk_a_text
            snippet_b = chunk_b_text[:150] + "..." if len(chunk_b_text) > 150 else chunk_b_text
            
            user_message = (
                f"I can't answer definitively because the retrieved sources conflict. "
                f"Source A ({first_pair.chunk_a['doc_id']}) says: {snippet_a} "
                f"While Source B ({first_pair.chunk_b['doc_id']}) says: {snippet_b} "
                f"Please confirm the authoritative policy or escalate to an owner."
            )
        else:
            user_message = (
                "I can't answer definitively because the retrieved sources contain conflicting information. "
                "Please confirm the authoritative policy or escalate to an owner."
            )
        
        return {
            "decision": "ABSTAIN",
            "reasons": ["conflict_detected", conflicts.conflict_type or "unknown"],
            "user_message": user_message,
            "thresholds": thresholds,
            "signals": signals,
            "risk": risk_result
        }
    
    # Decision precedence: C) Check retrieval quality, freshness, ambiguity
    # C1: Check retrieval confidence and hit count
    if (retrieval_quality.confidence.max < confidence_threshold or 
        retrieval_quality.confidence.hit_count < config.min_chunks):
        if retrieval_quality.confidence.max < confidence_threshold:
            reasons.append("low_retrieval_confidence")
        if retrieval_quality.confidence.hit_count < DEFAULT_MIN_CHUNKS:
            reasons.append("insufficient_retrieval_hits")
        
        user_message = "I don't have enough reliable evidence in the retrieved documents to answer this safely."
        
        logger.warning(
            f"ABSTAIN decision: low retrieval quality. "
            f"confidence={retrieval_quality.confidence.max:.4f} < {confidence_threshold}, "
            f"hits={retrieval_quality.confidence.hit_count} < {config.min_chunks}"
        )
        
        return {
            "decision": "ABSTAIN",
            "reasons": reasons,
            "user_message": user_message,
            "thresholds": thresholds,
            "signals": signals,
            "risk": risk_result
        }
    
    # C2: Check freshness violations (for high/medium risk)
    if retrieval_quality.freshness.freshness_violation and risk_level in ("high", "medium"):
        reasons.append("stale_documents")
        user_message = "The retrieved sources appear outdated, so I can't answer confidently."
        
        logger.warning(
            f"ABSTAIN decision: stale documents for {risk_level}-risk query. "
            f"Freshness violation count: {retrieval_quality.freshness.freshness_violation_count}"
        )
        
        return {
            "decision": "ABSTAIN",
            "reasons": reasons,
            "user_message": user_message,
            "thresholds": thresholds,
            "signals": signals,
            "risk": risk_result
        }
    
    # C3: Check for ambiguity
    is_amb, clarifying_q = is_ambiguous(query)
    if is_amb:
        logger.info(f"CLARIFY decision: ambiguous query. Question: {clarifying_q}")
        return {
            "decision": "CLARIFY",
            "reasons": ["ambiguous_query"],
            "user_message": clarifying_q or "Could you provide more context to clarify your question?",
            "thresholds": thresholds,
            "signals": signals,
            "risk": risk_result
        }
    
    # D) Otherwise, ANSWER
    logger.info(f"ANSWER decision: all checks passed. confidence={retrieval_quality.confidence.max:.4f}")
    return {
        "decision": "ANSWER",
        "reasons": [],
        "user_message": None,  # Will be filled by LLM
        "thresholds": thresholds,
        "signals": signals,
        "risk": risk_result
    }

