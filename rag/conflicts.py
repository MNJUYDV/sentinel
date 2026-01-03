"""
Conflict detection module for RAG safety system.
Detects contradictory policy statements in retrieved chunks.
"""
import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Polarity keywords for conflict detection
ALLOW_KEYWORDS = {
    "allowed", "permitted", "may", "can", "enable", "enables", "enabled",
    "refundable", "refunds allowed", "refund", "refunds"
}

PROHIBIT_KEYWORDS = {
    "prohibited", "must not", "may not", "cannot", "can not", "forbidden",
    "non-refundable", "no refunds", "not refundable", "refunds not allowed"
}

ENCRYPT_KEYWORDS = {
    "encrypt", "encrypted", "encryption", "must be encrypted", "encrypt at rest",
    "not plaintext", "must not be plaintext"
}

PLAINTEXT_KEYWORDS = {
    "plaintext", "store in plaintext", "stored in plaintext", "unencrypted"
}

# Topic keywords for relevance matching
TOPIC_KEYWORDS = {
    "ssn", "social security", "sensitive data", "customer data",
    "refund", "refunds", "purchase", "promotional",
    "rate limit", "rate limits", "requests per hour", "req/hr", "api rate",
    "encrypt", "encryption", "plaintext", "data storage"
}


def extract_topic_keywords(text: str) -> Set[str]:
    """Extract topic keywords from text (normalized to lowercase)."""
    text_lower = text.lower()
    found_topics = set()
    for topic in TOPIC_KEYWORDS:
        if topic in text_lower:
            found_topics.add(topic)
    return found_topics


def extract_polarity_keywords(text: str) -> Dict[str, Set[str]]:
    """Extract polarity keywords from text."""
    text_lower = text.lower()
    found = {
        "allow": set(),
        "prohibit": set(),
        "encrypt": set(),
        "plaintext": set()
    }
    
    for keyword in ALLOW_KEYWORDS:
        if keyword in text_lower:
            found["allow"].add(keyword)
    
    for keyword in PROHIBIT_KEYWORDS:
        if keyword in text_lower:
            found["prohibit"].add(keyword)
    
    for keyword in ENCRYPT_KEYWORDS:
        if keyword in text_lower:
            found["encrypt"].add(keyword)
    
    for keyword in PLAINTEXT_KEYWORDS:
        if keyword in text_lower:
            found["plaintext"].add(keyword)
    
    return found


def extract_numeric_values(text: str) -> List[Dict[str, any]]:
    """
    Extract numeric values with their units from text.
    Returns list of dicts with 'value', 'unit', and 'context' (surrounding text).
    """
    # Pattern to match numbers with optional units
    # Matches: "1000 requests per hour", "$50", "30 days", "100 req/hr", etc.
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z/]+(?:\s+[a-zA-Z]+)*)?'
    
    numeric_values = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        value_str = match.group(1)
        unit_str = match.group(2).strip() if match.group(2) else ""
        
        try:
            value = float(value_str)
            # Get context (20 chars before and after)
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].lower()
            
            numeric_values.append({
                "value": value,
                "unit": unit_str.lower() if unit_str else "",
                "context": context
            })
        except ValueError:
            continue
    
    return numeric_values


def check_polarity_conflict(chunk_a: dict, chunk_b: dict) -> Optional[Dict]:
    """Check for polarity conflicts (allowed vs prohibited)."""
    text_a = chunk_a["text"].lower()
    text_b = chunk_b["text"].lower()
    
    polarity_a = extract_polarity_keywords(text_a)
    polarity_b = extract_polarity_keywords(text_b)
    
    # Check for allow vs prohibit conflict
    if (polarity_a["allow"] and polarity_b["prohibit"]) or \
       (polarity_a["prohibit"] and polarity_b["allow"]):
        # Check topic overlap to avoid false positives
        topics_a = extract_topic_keywords(text_a)
        topics_b = extract_topic_keywords(text_b)
        if topics_a & topics_b:  # Intersection exists
            # Extract evidence snippets
            evidence_a = _extract_evidence_snippet(text_a, polarity_a["allow"] | polarity_a["prohibit"])
            evidence_b = _extract_evidence_snippet(text_b, polarity_b["allow"] | polarity_b["prohibit"])
            
            return {
                "chunk_a": {"doc_id": chunk_a["doc_id"], "chunk_id": chunk_a["chunk_id"]},
                "chunk_b": {"doc_id": chunk_b["doc_id"], "chunk_id": chunk_b["chunk_id"]},
                "reason": "must_not vs may / allowed vs prohibited",
                "evidence_snippets": {"a": evidence_a, "b": evidence_b}
            }
    
    # Check for refund conflicts
    if (polarity_a["allow"] and "refund" in text_a) and \
       (polarity_b["prohibit"] and "refund" in text_b):
        topics_a = extract_topic_keywords(text_a)
        topics_b = extract_topic_keywords(text_b)
        if "refund" in topics_a and "refund" in topics_b:
            evidence_a = _extract_evidence_snippet(text_a, {"refund"})
            evidence_b = _extract_evidence_snippet(text_b, {"refund"})
            
            return {
                "chunk_a": {"doc_id": chunk_a["doc_id"], "chunk_id": chunk_a["chunk_id"]},
                "chunk_b": {"doc_id": chunk_b["doc_id"], "chunk_id": chunk_b["chunk_id"]},
                "reason": "refundable vs non-refundable",
                "evidence_snippets": {"a": evidence_a, "b": evidence_b}
            }
    
    # Check for plaintext vs encrypt conflict
    if (polarity_a["plaintext"] and polarity_b["encrypt"]) or \
       (polarity_a["encrypt"] and polarity_b["plaintext"]):
        topics_a = extract_topic_keywords(text_a)
        topics_b = extract_topic_keywords(text_b)
        if (topics_a & topics_b) or ("ssn" in text_a and "ssn" in text_b):
            evidence_a = _extract_evidence_snippet(text_a, polarity_a["plaintext"] | polarity_a["encrypt"])
            evidence_b = _extract_evidence_snippet(text_b, polarity_b["plaintext"] | polarity_b["encrypt"])
            
            return {
                "chunk_a": {"doc_id": chunk_a["doc_id"], "chunk_id": chunk_a["chunk_id"]},
                "chunk_b": {"doc_id": chunk_b["doc_id"], "chunk_id": chunk_b["chunk_id"]},
                "reason": "store in plaintext vs must be encrypted / not plaintext",
                "evidence_snippets": {"a": evidence_a, "b": evidence_b}
            }
    
    return None


def check_numeric_conflict(chunk_a: dict, chunk_b: dict) -> Optional[Dict]:
    """Check for numeric conflicts (same unit, different values)."""
    text_a = chunk_a["text"]
    text_b = chunk_b["text"]
    
    numeric_a = extract_numeric_values(text_a)
    numeric_b = extract_numeric_values(text_b)
    
    # Check for same unit but different values
    for num_a in numeric_a:
        for num_b in numeric_b:
            # Check if units are similar (same unit tokens)
            if num_a["unit"] and num_b["unit"]:
                # Normalize units: "requests per hour" -> "req/hr", "req/hr" -> "req/hr"
                unit_a_norm = _normalize_unit(num_a["unit"])
                unit_b_norm = _normalize_unit(num_b["unit"])
                
                # Check if units match (fuzzy match for common variations)
                if unit_a_norm == unit_b_norm and num_a["value"] != num_b["value"]:
                    # Check topic overlap
                    topics_a = extract_topic_keywords(text_a)
                    topics_b = extract_topic_keywords(text_b)
                    if topics_a & topics_b:
                        evidence_a = _extract_numeric_evidence(text_a, num_a["value"], num_a["unit"])
                        evidence_b = _extract_numeric_evidence(text_b, num_b["value"], num_b["unit"])
                        
                        return {
                            "chunk_a": {"doc_id": chunk_a["doc_id"], "chunk_id": chunk_a["chunk_id"]},
                            "chunk_b": {"doc_id": chunk_b["doc_id"], "chunk_id": chunk_b["chunk_id"]},
                            "reason": f"numeric conflict: {num_a['value']} vs {num_b['value']} {unit_a_norm}",
                            "evidence_snippets": {"a": evidence_a, "b": evidence_b}
                        }
    
    return None


def _normalize_unit(unit: str) -> str:
    """Normalize unit strings for comparison."""
    unit = unit.lower().strip()
    
    # Common normalizations
    if "request" in unit and ("hour" in unit or "hr" in unit):
        return "req/hr"
    if "req" in unit and ("hour" in unit or "hr" in unit):
        return "req/hr"
    if "per hour" in unit or "/hr" in unit:
        return "req/hr"
    
    if "day" in unit:
        return "days"
    if "dollar" in unit or unit.startswith("$"):
        return "$"
    
    return unit


def _extract_evidence_snippet(text: str, keywords: Set[str]) -> str:
    """Extract a snippet containing the relevant keywords."""
    text_lower = text.lower()
    
    # Find positions of keywords
    positions = []
    for keyword in keywords:
        idx = text_lower.find(keyword)
        if idx != -1:
            positions.append((idx, idx + len(keyword)))
    
    if not positions:
        # Fallback: return first 100 chars
        return text[:100] + "..." if len(text) > 100 else text
    
    # Get min start and max end
    min_start = min(p[0] for p in positions)
    max_end = max(p[1] for p in positions)
    
    # Expand context (50 chars before and after)
    start = max(0, min_start - 50)
    end = min(len(text), max_end + 50)
    
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet


def _extract_numeric_evidence(text: str, value: float, unit: str) -> str:
    """Extract a snippet containing the numeric value and unit."""
    # Find the position of the value in text
    value_str = str(int(value)) if value.is_integer() else str(value)
    idx = text.lower().find(value_str)
    
    if idx == -1:
        return text[:100] + "..." if len(text) > 100 else text
    
    # Expand context
    start = max(0, idx - 50)
    end = min(len(text), idx + len(value_str) + len(unit) + 50)
    
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet


def detect_conflicts(retrieved_chunks: List[dict]) -> Dict:
    """
    Detect conflicts in retrieved chunks.
    
    Args:
        retrieved_chunks: List of chunk dictionaries with doc_id, chunk_id, text
    
    Returns:
        Dict with conflict_detected, conflict_type, pairs, summary
    """
    if len(retrieved_chunks) < 2:
        return {
            "conflict_detected": False,
            "conflict_type": None,
            "pairs": [],
            "summary": "No conflicts detected (insufficient chunks for comparison)"
        }
    
    conflict_pairs = []
    
    # Check all pairs of chunks
    for i in range(len(retrieved_chunks)):
        for j in range(i + 1, len(retrieved_chunks)):
            chunk_a = retrieved_chunks[i]
            chunk_b = retrieved_chunks[j]
            
            # Skip if same document (conflicts should be cross-document)
            if chunk_a["doc_id"] == chunk_b["doc_id"]:
                continue
            
            # Check polarity conflicts first (most common)
            polarity_conflict = check_polarity_conflict(chunk_a, chunk_b)
            if polarity_conflict:
                conflict_pairs.append({
                    **polarity_conflict,
                    "conflict_type": "policy"
                })
                continue
            
            # Check numeric conflicts
            numeric_conflict = check_numeric_conflict(chunk_a, chunk_b)
            if numeric_conflict:
                conflict_pairs.append({
                    **numeric_conflict,
                    "conflict_type": "numeric"
                })
                continue
    
    if conflict_pairs:
        # Determine primary conflict type
        conflict_types = [pair["conflict_type"] for pair in conflict_pairs]
        primary_type = conflict_types[0] if conflict_types else "other"
        
        # Generate summary
        if primary_type == "policy":
            summary = f"Policy conflict detected: {len(conflict_pairs)} conflicting pair(s) found"
        elif primary_type == "numeric":
            summary = f"Numeric conflict detected: {len(conflict_pairs)} conflicting pair(s) found"
        else:
            summary = f"Conflict detected: {len(conflict_pairs)} conflicting pair(s) found"
        
        # Log conflicts
        doc_ids_involved = set()
        for pair in conflict_pairs:
            doc_ids_involved.add(pair["chunk_a"]["doc_id"])
            doc_ids_involved.add(pair["chunk_b"]["doc_id"])
        
        logger.warning(
            f"Conflict detected: type={primary_type}, pairs={len(conflict_pairs)}, "
            f"doc_ids={sorted(doc_ids_involved)}"
        )
        
        return {
            "conflict_detected": True,
            "conflict_type": primary_type,
            "pairs": conflict_pairs,
            "summary": summary
        }
    else:
        return {
            "conflict_detected": False,
            "conflict_type": None,
            "pairs": [],
            "summary": "No conflicts detected"
        }

