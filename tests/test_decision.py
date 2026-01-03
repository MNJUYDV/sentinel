"""
Tests for decision engine (Day 5).
"""
import pytest
from datetime import datetime, timedelta

from rag.config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK,
    DEFAULT_MIN_CHUNKS
)
from rag.decision import decide, is_ambiguous
from rag.risk import classify_risk
from rag.schemas import ConflictResult, RetrievalConfidence, RetrievalFreshness, RetrievalQuality, ValidationResult


def make_chunk(doc_id="doc1", chunk_id="chunk1", similarity=0.8, days_old=0):
    """Helper to create a chunk dict with timestamp."""
    timestamp = (datetime.now() - timedelta(days=days_old)).isoformat()
    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "timestamp": timestamp,
        "similarity": similarity,
        "text": f"Sample text from {doc_id} {chunk_id}"
    }


def make_retrieval_quality(confidence_max=0.8, hit_count=3, freshness_violation=False, freshness_days=90):
    """Helper to create RetrievalQuality object."""
    return RetrievalQuality(
        confidence=RetrievalConfidence(
            max=confidence_max,
            mean=confidence_max - 0.1,
            gap=0.1,
            hit_count=hit_count
        ),
        freshness=RetrievalFreshness(
            oldest_timestamp=datetime.now().isoformat(),
            newest_timestamp=datetime.now().isoformat(),
            freshness_violation_count=1 if freshness_violation else 0,
            freshness_violation=freshness_violation,
            freshness_days=freshness_days
        ),
        top_doc_ids=["doc1", "doc2"],
        top_timestamps=[datetime.now().isoformat(), datetime.now().isoformat()]
    )


class TestRiskClassification:
    """Test risk classification."""
    
    def test_high_risk_keywords(self):
        """Test high-risk keyword detection."""
        result = classify_risk("What is our SSN encryption policy?")
        assert result["risk_level"] == "high"
        assert len(result["matched_keywords"]) > 0
        assert any("ssn" in kw.lower() or "encrypt" in kw.lower() for kw in result["matched_keywords"])
    
    def test_medium_risk_keywords(self):
        """Test medium-risk keyword detection."""
        result = classify_risk("What are our rate limits?")
        assert result["risk_level"] == "medium"
        assert "rate limit" in " ".join(result["matched_keywords"]).lower()
    
    def test_low_risk(self):
        """Test low-risk classification."""
        result = classify_risk("How do I use the API?")
        assert result["risk_level"] == "low"
        assert len(result["matched_keywords"]) == 0


class TestAmbiguityDetection:
    """Test ambiguity detection."""
    
    def test_ambiguous_permission_query(self):
        """Test detection of ambiguous permission queries."""
        is_amb, question = is_ambiguous("Can we downgrade?")
        assert is_amb is True
        assert question is not None
        assert "plan" in question.lower() or "region" in question.lower()
    
    def test_clear_permission_query(self):
        """Test that queries with scope are not ambiguous."""
        is_amb, question = is_ambiguous("Can we downgrade the premium plan?")
        assert is_amb is False
    
    def test_ambiguous_temporal_query(self):
        """Test detection of ambiguous temporal queries."""
        is_amb, question = is_ambiguous("What is the current pricing?")
        assert is_amb is True
        assert question is not None
    
    def test_clear_temporal_query(self):
        """Test that queries with dates are not ambiguous."""
        is_amb, question = is_ambiguous("What was the pricing in January 2024?")
        assert is_amb is False


class TestDecisionEngine:
    """Test decision engine."""
    
    def test_block_on_invalid_citations(self):
        """Test BLOCK decision when citations are invalid."""
        chunks = [make_chunk()]
        retrieval_quality = make_retrieval_quality(confidence_max=0.9, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(
            citation_valid=False,
            errors=["Citation 'bad_citation' does not match any retrieved chunk"],
            warnings=[]
        )
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "BLOCK"
        assert "invalid_citations" in result["reasons"]
        assert result["user_message"] is not None
    
    def test_abstain_on_conflicts(self):
        """Test ABSTAIN decision when conflicts are detected."""
        chunks = [make_chunk("doc1", "chunk1"), make_chunk("doc2", "chunk2")]
        retrieval_quality = make_retrieval_quality(confidence_max=0.9, hit_count=2)
        conflicts = ConflictResult(
            conflict_detected=True,
            conflict_type="policy",
            pairs=[],
            summary="Conflict detected"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "conflict_detected" in result["reasons"]
        assert result["user_message"] is not None
    
    def test_abstain_low_confidence(self):
        """Test ABSTAIN decision when retrieval confidence is low."""
        chunks = [make_chunk(similarity=0.5)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.5, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "low_retrieval_confidence" in result["reasons"]
        assert result["user_message"] is not None
    
    def test_abstain_high_risk_low_confidence(self):
        """Test ABSTAIN decision for high-risk query with confidence below high-risk threshold."""
        chunks = [make_chunk(similarity=0.65)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.65, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="What is our SSN encryption policy?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "low_retrieval_confidence" in result["reasons"]
        assert result["risk"]["risk_level"] == "high"
        assert result["thresholds"]["conf_max"] == DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK
    
    def test_abstain_insufficient_hits(self):
        """Test ABSTAIN decision when hit count is below minimum."""
        chunks = [make_chunk()]
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, hit_count=1)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "insufficient_retrieval_hits" in result["reasons"]
    
    def test_abstain_stale_documents_high_risk(self):
        """Test ABSTAIN decision for stale documents on high-risk query."""
        chunks = [make_chunk(days_old=100)]
        retrieval_quality = make_retrieval_quality(
            confidence_max=0.8,
            hit_count=3,
            freshness_violation=True,
            freshness_days=30
        )
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="What is our refund policy?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "stale_documents" in result["reasons"]
        assert result["risk"]["risk_level"] == "high"
    
    def test_abstain_stale_documents_medium_risk(self):
        """Test ABSTAIN decision for stale documents on medium-risk query."""
        chunks = [make_chunk(days_old=100)]
        retrieval_quality = make_retrieval_quality(
            confidence_max=0.8,
            hit_count=3,
            freshness_violation=True,
            freshness_days=90
        )
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="What are our rate limits?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "stale_documents" in result["reasons"]
        assert result["risk"]["risk_level"] == "medium"
    
    def test_clarify_ambiguous_query(self):
        """Test CLARIFY decision for ambiguous queries."""
        chunks = [make_chunk(similarity=0.8)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="Can we downgrade without penalties?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "CLARIFY"
        assert "ambiguous_query" in result["reasons"]
        assert result["user_message"] is not None
        assert "clarify" in result["user_message"].lower() or "?" in result["user_message"]
    
    def test_answer_high_confidence_low_risk(self):
        """Test ANSWER decision for high-confidence, low-risk query."""
        chunks = [make_chunk(similarity=0.8), make_chunk(similarity=0.75)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, hit_count=2)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="How do I use the API?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ANSWER"
        assert len(result["reasons"]) == 0
        assert result["user_message"] is None
        assert result["risk"]["risk_level"] == "low"
    
    def test_answer_high_confidence_high_risk(self):
        """Test ANSWER decision for high-confidence, high-risk query that meets threshold."""
        chunks = [make_chunk(similarity=0.75), make_chunk(similarity=0.72)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.75, hit_count=2)
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="What is our SSN encryption policy?",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ANSWER"
        assert result["risk"]["risk_level"] == "high"
        assert result["thresholds"]["conf_max"] == DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK


class TestDecisionPrecedence:
    """Test that decision precedence is correct."""
    
    def test_block_takes_precedence_over_conflicts(self):
        """Test that BLOCK (invalid citations) takes precedence over conflicts."""
        chunks = [make_chunk()]
        retrieval_quality = make_retrieval_quality(confidence_max=0.9, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=True,
            conflict_type="policy",
            pairs=[],
            summary="Conflict detected"
        )
        validation = ValidationResult(
            citation_valid=False,
            errors=["Invalid citation"],
            warnings=[]
        )
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "BLOCK"
    
    def test_conflicts_take_precedence_over_quality(self):
        """Test that conflicts take precedence over retrieval quality issues."""
        chunks = [make_chunk(similarity=0.5)]
        retrieval_quality = make_retrieval_quality(confidence_max=0.5, hit_count=3)
        conflicts = ConflictResult(
            conflict_detected=True,
            conflict_type="policy",
            pairs=[],
            summary="Conflict detected"
        )
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        
        result = decide(
            query="Test query",
            retrieved_chunks=chunks,
            retrieval_quality=retrieval_quality,
            conflicts=conflicts,
            validation=validation
        )
        
        assert result["decision"] == "ABSTAIN"
        assert "conflict_detected" in result["reasons"]

