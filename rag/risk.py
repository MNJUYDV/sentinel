"""
Risk classification module for RAG safety system.
Classifies queries by risk level based on keywords.
"""
import logging
from typing import Dict, List, Set

logger = logging.getLogger(__name__)

# High-risk keywords (legal, compliance, security, financial)
HIGH_RISK_KEYWORDS = {
    "policy", "policies", "legal", "compliance", "security", "ssn", "ssns",
    "pii", "encryption", "encrypt", "soc2", "hipaa", "refund", "refunds",
    "chargeback", "chargebacks", "pricing", "limits", "limit", "retention",
    "delete", "gdpr", "personally identifiable", "sensitive data"
}

# Medium-risk keywords (operational but not legal)
MEDIUM_RISK_KEYWORDS = {
    "rate limit", "rate limits", "sla", "slas", "uptime", "quota", "quotas",
    "throttle", "throttling", "bandwidth", "capacity"
}


def classify_risk(query: str) -> Dict[str, any]:
    """
    Classify query risk level based on keywords.
    
    Args:
        query: User query string
    
    Returns:
        Dict with:
        - risk_level: "high" | "medium" | "low"
        - matched_keywords: List of matched keywords for debugging
    """
    query_lower = query.lower()
    
    # Find matching keywords
    matched_high = {kw for kw in HIGH_RISK_KEYWORDS if kw in query_lower}
    matched_medium = {kw for kw in MEDIUM_RISK_KEYWORDS if kw in query_lower}
    
    # Determine risk level (high takes precedence)
    if matched_high:
        risk_level = "high"
        matched_keywords = list(matched_high)
    elif matched_medium:
        risk_level = "medium"
        matched_keywords = list(matched_medium)
    else:
        risk_level = "low"
        matched_keywords = []
    
    logger.debug(f"Risk classification: {risk_level}, keywords: {matched_keywords}")
    
    return {
        "risk_level": risk_level,
        "matched_keywords": matched_keywords
    }

