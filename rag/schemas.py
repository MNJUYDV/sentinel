"""
Pydantic schemas for RAG API requests and responses.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

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


class ValidationResult(BaseModel):
    """Citation validation results."""
    citation_valid: bool = Field(..., description="True if citations are valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings (non-blocking)")


class ConflictPair(BaseModel):
    """A pair of conflicting chunks."""
    chunk_a: Dict[str, str] = Field(..., description="First conflicting chunk (doc_id, chunk_id)")
    chunk_b: Dict[str, str] = Field(..., description="Second conflicting chunk (doc_id, chunk_id)")
    reason: str = Field(..., description="Reason for conflict")
    evidence_snippets: Dict[str, str] = Field(..., description="Evidence snippets from both chunks")
    conflict_type: str = Field(..., description="Type of conflict: policy, numeric, temporal, other")


class ConflictResult(BaseModel):
    """Result of conflict detection."""
    conflict_detected: bool = Field(..., description="True if conflicts were detected")
    conflict_type: Optional[str] = Field(None, description="Type of conflict: policy, numeric, temporal, other")
    pairs: List[ConflictPair] = Field(default_factory=list, description="List of conflicting chunk pairs")
    summary: str = Field(..., description="Human-readable summary of conflict detection")


class RiskResult(BaseModel):
    """Risk classification result."""
    risk_level: str = Field(..., description="Risk level: high, medium, or low")
    matched_keywords: List[str] = Field(default_factory=list, description="Keywords that matched for risk classification")


class DecisionResult(BaseModel):
    """Result of decision engine."""
    decision: str = Field(..., description="Decision: ANSWER, ANSWER_WITH_CAVEATS, CLARIFY, ABSTAIN, or BLOCK")
    reasons: List[str] = Field(default_factory=list, description="List of reasons for the decision")
    user_message: Optional[str] = Field(None, description="Message to show user (if not ANSWER)")
    thresholds: Dict[str, Any] = Field(default_factory=dict, description="Threshold values used for decision")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Signal values (confidence, freshness, etc.)")
    risk: RiskResult = Field(..., description="Risk classification result")


class AnswerRequest(BaseModel):
    """Request to answer a query using RAG."""
    query: str = Field(..., min_length=1, description="User query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    freshness_days: Optional[int] = Field(
        None, 
        ge=0, 
        description="Override freshness threshold in days (optional: >= 1 for threshold, 0 to disable, None for default)"
    )


class AnswerResponse(BaseModel):
    """Response containing answer and retrieval results with citation validation, conflict detection, and decision engine."""
    query: str = Field(..., description="Original query")
    decision: str = Field(..., description="ANSWER, ANSWER_WITH_CAVEATS, CLARIFY, ABSTAIN, or BLOCK")
    answer: str = Field(..., description="Generated answer, safe fallback if blocked, clarification request, or abstention message")
    citations: List[str] = Field(..., description="List of citation IDs")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")
    validation: ValidationResult = Field(..., description="Citation validation results")
    conflicts: ConflictResult = Field(..., description="Conflict detection results")
    reasons: List[str] = Field(default_factory=list, description="Reasons for the decision")
    signals: Dict[str, Any] = Field(default_factory=dict, description="Decision signals (confidence, freshness, etc.)")


class DebugRetrievalResponse(BaseModel):
    """Response for debug retrieval endpoint (no LLM call)."""
    query: str = Field(..., description="Original query")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")


class DebugConflictsResponse(BaseModel):
    """Response for debug conflicts endpoint (retrieval + conflict detection only)."""
    query: str = Field(..., description="Original query")
    retrieval: RetrievalResult = Field(..., description="Retrieval results")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")
    conflicts: ConflictResult = Field(..., description="Conflict detection results")


class DebugDecisionResponse(BaseModel):
    """Response for debug decision endpoint (retrieval + conflict detection + decision engine, no LLM call)."""
    query: str = Field(..., description="Original query")
    risk: RiskResult = Field(..., description="Risk classification result")
    retrieval_quality: RetrievalQuality = Field(..., description="Retrieval quality signals")
    conflicts: ConflictResult = Field(..., description="Conflict detection results")
    validation: Optional[ValidationResult] = Field(None, description="Citation validation results (if applicable)")
    decision_result: DecisionResult = Field(..., description="Decision engine result")


class ValidateRequest(BaseModel):
    """Request to validate citations."""
    answer: str = Field(..., description="Answer text")
    citations: List[str] = Field(..., description="List of citation IDs")
    retrieved_chunks: List[ChunkInfo] = Field(..., description="Retrieved chunks to validate against")


class ValidateResponse(BaseModel):
    """Response from citation validation."""
    citation_valid: bool = Field(..., description="True if citations are valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class DocumentInfo(BaseModel):
    """Information about a loaded document."""
    doc_id: str
    title: str
    timestamp: str


class DocsResponse(BaseModel):
    """Response containing loaded documents information."""
    count: int = Field(..., description="Number of loaded documents")
    documents: List[DocumentInfo] = Field(..., description="List of document info")
