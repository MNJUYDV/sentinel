"""
Retrieval logic for finding relevant chunks and computing quality signals.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from rag.config import DEFAULT_FRESHNESS_DAYS
from rag.ingest import DocumentChunk, VectorIndex
from rag.schemas import RetrievalConfidence, RetrievalFreshness, RetrievalQuality

logger = logging.getLogger(__name__)


def retrieve(
    index: VectorIndex,
    query: str,
    top_k: int = 5
) -> List[dict]:
    """
    Retrieve top-k most relevant chunks for a query.
    
    Args:
        index: VectorIndex instance
        query: Search query
        top_k: Number of chunks to retrieve
    
    Returns:
        List of chunk dictionaries with doc_id, chunk_id, timestamp, similarity, text
    """
    logger.info(f"Retrieving top {top_k} chunks for query: {query[:50]}...")
    
    # Search index
    results = index.search(query, top_k)
    
    # Format results
    retrieved_chunks = []
    for chunk, similarity in results:
        retrieved_chunks.append({
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "timestamp": chunk.timestamp,
            "similarity": round(similarity, 4),
            "text": chunk.text
        })
    
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
    return retrieved_chunks


def compute_retrieval_quality(
    chunks: List[dict],
    freshness_days: Optional[int] = None,
    disable_freshness: bool = False
) -> RetrievalQuality:
    """
    Compute retrieval quality signals including confidence and freshness metrics.
    
    Args:
        chunks: List of retrieved chunk dictionaries
        freshness_days: Freshness threshold in days. None uses DEFAULT_FRESHNESS_DAYS.
        disable_freshness: If True, skip freshness violation calculation (treat as disabled)
    
    Returns:
        RetrievalQuality object with all computed signals
    """
    # Handle freshness_days: None means use default, unless freshness is disabled
    if disable_freshness:
        effective_freshness_days = 0  # 0 indicates disabled
    else:
        effective_freshness_days = DEFAULT_FRESHNESS_DAYS if freshness_days is None else freshness_days
    
    if not chunks:
        # Return empty quality signals if no chunks
        return RetrievalQuality(
            confidence=RetrievalConfidence(
                max=0.0,
                mean=0.0,
                gap=None,
                hit_count=0
            ),
            freshness=RetrievalFreshness(
                oldest_timestamp="",
                newest_timestamp="",
                freshness_violation_count=0,
                freshness_violation=False,
                freshness_days=effective_freshness_days
            ),
            top_doc_ids=[],
            top_timestamps=[]
        )
    
    # Compute confidence signals
    similarities = [chunk["similarity"] for chunk in chunks]
    max_similarity = max(similarities)
    mean_similarity = sum(similarities) / len(similarities)
    hit_count = len(chunks)
    
    # Compute gap (top1 - top2) if we have at least 2 chunks
    gap = None
    if len(similarities) >= 2:
        gap = round(similarities[0] - similarities[1], 4)
    
    confidence = RetrievalConfidence(
        max=round(max_similarity, 4),
        mean=round(mean_similarity, 4),
        gap=gap,
        hit_count=hit_count
    )
    
    # Compute freshness signals
    timestamps = [chunk["timestamp"] for chunk in chunks]
    
    # Parse timestamps and find oldest/newest
    parsed_timestamps = []
    for ts in timestamps:
        try:
            # Parse ISO format timestamp
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            parsed_timestamps.append(dt)
        except (ValueError, AttributeError):
            # If parsing fails, skip this timestamp for date calculations
            logger.warning(f"Could not parse timestamp: {ts}")
    
    oldest_timestamp = min(timestamps) if timestamps else ""
    newest_timestamp = max(timestamps) if timestamps else ""
    
    # Compute freshness violations
    violation_count = 0
    freshness_days_for_response = effective_freshness_days
    
    # Skip freshness violation calculation if:
    # 1. No timestamps to check
    # 2. freshness_days is 0 (disabled)
    # 3. freshness_days would cause overflow (Python datetime has limits ~36500 days is safe)
    MAX_SAFE_FRESHNESS_DAYS = 36500  # ~100 years, safe limit for datetime
    
    if not disable_freshness and parsed_timestamps and effective_freshness_days > 0 and effective_freshness_days <= MAX_SAFE_FRESHNESS_DAYS:
        try:
            threshold_date = datetime.now().replace(tzinfo=parsed_timestamps[0].tzinfo) - timedelta(days=effective_freshness_days)
            violation_count = sum(1 for dt in parsed_timestamps if dt < threshold_date)
        except (OverflowError, OSError):
            # If overflow occurs (shouldn't happen with our check, but be safe), skip violations
            violation_count = 0
    # If freshness is disabled or effective_freshness_days is 0, violation_count stays 0
    
    freshness = RetrievalFreshness(
        oldest_timestamp=oldest_timestamp,
        newest_timestamp=newest_timestamp,
        freshness_violation_count=violation_count,
        freshness_violation=violation_count > 0,
        freshness_days=freshness_days_for_response
    )
    
    # Extract top doc_ids and timestamps
    top_doc_ids = [chunk["doc_id"] for chunk in chunks]
    top_timestamps = timestamps
    
    return RetrievalQuality(
        confidence=confidence,
        freshness=freshness,
        top_doc_ids=top_doc_ids,
        top_timestamps=top_timestamps
    )
