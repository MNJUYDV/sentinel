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


class AnswerRequest(BaseModel):
    """Request to answer a query using RAG."""
    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class AnswerResponse(BaseModel):
    """Response containing answer and retrieval results."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")


class DocumentInfo(BaseModel):
    """Information about a loaded document."""
    doc_id: str
    title: str
    timestamp: str


class DocsResponse(BaseModel):
    """Response containing loaded documents information."""
    count: int = Field(..., description="Number of loaded documents")
    documents: List[DocumentInfo] = Field(..., description="List of document info")

