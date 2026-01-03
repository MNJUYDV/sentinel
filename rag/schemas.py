"""
Pydantic schemas for RAG API requests and responses.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkInfo(BaseModel):
    """Information about a retrieved chunk."""
    doc_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID within the document")
    timestamp: str = Field(..., description="ISO timestamp of the document")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    text: str = Field(..., description="Chunk text content")


class RetrievalResult(BaseModel):
    """Result of retrieval operation."""
    top_k: int = Field(..., description="Number of chunks requested")
    chunks: List[ChunkInfo] = Field(..., description="Retrieved chunks sorted by similarity")


class RetrievalConfidence(BaseModel):
    """Retrieval confidence signals."""
    max: float = Field(..., ge=0.0, le=1.0, description="Maximum similarity among retrieved chunks")
    mean: float = Field(..., ge=0.0, le=1.0, description="Mean similarity among retrieved chunks")
    gap: Optional[float] = Field(None, description="Difference between top1 and top2 similarity (null if top_k < 2)")
    hit_count: int = Field(..., ge=0, description="Number of retrieved chunks returned")


class RetrievalFreshness(BaseModel):
    """Freshness/staleness signals for retrieved chunks."""
    oldest_timestamp: str = Field(..., description="Oldest timestamp (ISO date) among retrieved chunks")
    newest_timestamp: str = Field(..., description="Newest timestamp (ISO date) among retrieved chunks")
    freshness_violation_count: int = Field(..., ge=0, description="Count of chunks older than freshness threshold")
    freshness_violation: bool = Field(..., description="True if any chunks violate freshness threshold")
    freshness_days: int = Field(..., description="Freshness threshold in days used for this retrieval")


class RetrievalQuality(BaseModel):
    """Overall retrieval quality signals."""
    confidence: RetrievalConfidence = Field(..., description="Retrieval confidence signals")
    freshness: RetrievalFreshness = Field(..., description="Freshness/staleness signals")
    top_doc_ids: List[str] = Field(..., description="Document IDs of retrieved chunks (in order)")
    top_timestamps: List[str] = Field(..., description="Timestamps of retrieved chunks (in order)")


class AnswerRequest(BaseModel):
    """Request to answer a query using RAG."""
    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    freshness_days: Optional[int] = Field(None, ge=1, description="Override freshness threshold in days (optional)")


class AnswerResponse(BaseModel):
    """Response containing answer and retrieval results."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")


class DebugRetrievalResponse(BaseModel):
    """Response for debug retrieval endpoint (no LLM call)."""
    query: str = Field(..., description="Original query")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")


class DocumentInfo(BaseModel):
    """Information about a loaded document."""
    doc_id: str
    title: str
    timestamp: str


class DocsResponse(BaseModel):
    """Response containing loaded documents information."""
    count: int = Field(..., description="Number of loaded documents")
    documents: List[DocumentInfo] = Field(..., description="List of document info")
