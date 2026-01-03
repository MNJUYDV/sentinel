"""
Retrieval logic for finding relevant chunks.
"""
import logging
from typing import List

from rag.ingest import DocumentChunk, VectorIndex

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

