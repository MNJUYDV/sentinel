"""
Citation validation module for RAG safety system.
Validates that citations reference actual retrieved chunks.
"""
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Common English stopwords for keyword overlap check
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
    'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
    'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
    'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get',
    'come', 'made', 'may', 'part'
}

MAX_CITATIONS = 5


def is_insufficiency_message(answer: str) -> bool:
    """
    Check if answer indicates insufficient context/refusal.
    Uses keyword matching for common insufficiency patterns.
    """
    answer_lower = answer.lower()
    insufficiency_indicators = [
        'insufficient',
        'not enough information',
        "don't have enough",
        'do not have enough',
        "can't provide",
        'cannot provide',
        'unable to answer',
        'not sure',
        'not certain',
        'unclear',
        'cannot determine',
        'cannot find',
        'not found in',
        'not available in',
        'not mentioned in',
        'no information about',
        'lack of context',
        'missing context'
    ]
    return any(indicator in answer_lower for indicator in insufficiency_indicators)


def extract_keywords(text: str) -> set:
    """
    Extract meaningful keywords from text, excluding stopwords.
    Returns lowercase keywords.
    """
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text.lower())
    # Filter stopwords and short words (< 3 chars)
    keywords = {w for w in words if w not in STOPWORDS and len(w) >= 3}
    return keywords


def check_citation_keyword_overlap(answer: str, chunk_text: str) -> bool:
    """
    Check if at least one keyword from answer appears in chunk text.
    Returns True if overlap found, False otherwise.
    """
    answer_keywords = extract_keywords(answer)
    chunk_keywords = extract_keywords(chunk_text)
    
    # Check for overlap
    overlap = answer_keywords & chunk_keywords
    return len(overlap) > 0


def validate_citations(
    citations: List[str],
    retrieved_chunks: List[dict],
    answer: str
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate citations against retrieved chunks.
    
    Args:
        citations: List of citation strings (expected format: "doc_id#chunk_id")
        retrieved_chunks: List of retrieved chunk dictionaries
        answer: The answer text to check for keyword overlap
    
    Returns:
        Tuple of (is_valid: bool, errors: List[str], warnings: List[str])
    """
    errors = []
    warnings = []
    
    # Build set of valid chunk identifiers
    valid_chunk_ids = {chunk["chunk_id"] for chunk in retrieved_chunks}
    
    # Rule a: Every citation must exactly match a retrieved chunk identifier
    invalid_citations = []
    for citation in citations:
        if citation not in valid_chunk_ids:
            invalid_citations.append(citation)
            errors.append(f"Citation '{citation}' does not match any retrieved chunk")
    
    if invalid_citations:
        return (False, errors, warnings)
    
    # Rule b: No duplicates
    if len(citations) != len(set(citations)):
        duplicates = [c for c in citations if citations.count(c) > 1]
        errors.append(f"Duplicate citations found: {set(duplicates)}")
        return (False, errors, warnings)
    
    # Rule c: Limit to max 5 citations
    if len(citations) > MAX_CITATIONS:
        errors.append(f"Too many citations ({len(citations)}), maximum is {MAX_CITATIONS}")
        return (False, errors, warnings)
    
    # Rule d: If citations empty AND answer is not insufficiency, mark invalid
    if not citations and not is_insufficiency_message(answer):
        errors.append("Citations are empty but answer does not indicate insufficient context")
        return (False, errors, warnings)
    
    # Rule e: Check keyword overlap (warning only, not blocking)
    if citations and answer and not is_insufficiency_message(answer):
        # Check if any cited chunk has keyword overlap
        overlap_found = False
        for citation in citations:
            # Find the chunk with this citation
            chunk = next((c for c in retrieved_chunks if c["chunk_id"] == citation), None)
            if chunk:
                if check_citation_keyword_overlap(answer, chunk["text"]):
                    overlap_found = True
                    break
        
        if not overlap_found:
            warnings.append(
                "Citation keyword overlap heuristic: No keywords from answer found in cited chunks. "
                "Citations may not directly support the answer."
            )
    
    # All validation rules passed
    return (True, errors, warnings)

