"""
Unit tests for retrieval quality signal computation.
"""
from datetime import datetime, timedelta

import pytest

from rag.retrieve import compute_retrieval_quality


def test_freshness_violation_count():
    """Test that freshness_violation_count is computed correctly given known timestamps."""
    # Create test chunks with timestamps
    now = datetime.now()
    
    # Chunk older than 90 days
    old_date = (now - timedelta(days=120)).isoformat() + "Z"
    # Chunk within 90 days
    recent_date = (now - timedelta(days=30)).isoformat() + "Z"
    # Very old chunk
    very_old_date = (now - timedelta(days=200)).isoformat() + "Z"
    
    chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": recent_date,
            "similarity": 0.9,
            "text": "Recent content"
        },
        {
            "doc_id": "doc2",
            "chunk_id": "doc2#p0",
            "timestamp": old_date,
            "similarity": 0.8,
            "text": "Old content"
        },
        {
            "doc_id": "doc3",
            "chunk_id": "doc3#p0",
            "timestamp": very_old_date,
            "similarity": 0.7,
            "text": "Very old content"
        }
    ]
    
    # Test with 90-day threshold (default)
    quality = compute_retrieval_quality(chunks, freshness_days=90)
    
    # Should have 2 violations (old_date and very_old_date are > 90 days old)
    assert quality.freshness.freshness_violation_count == 2
    assert quality.freshness.freshness_violation is True
    
    # Test with 150-day threshold (should only have 1 violation)
    quality_150 = compute_retrieval_quality(chunks, freshness_days=150)
    assert quality_150.freshness.freshness_violation_count == 1
    assert quality_150.freshness.freshness_violation is True
    
    # Test with 250-day threshold (no violations)
    quality_250 = compute_retrieval_quality(chunks, freshness_days=250)
    assert quality_250.freshness.freshness_violation_count == 0
    assert quality_250.freshness.freshness_violation is False


def test_retrieval_confidence_gap():
    """Test that retrieval_confidence_gap is computed correctly."""
    chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "High similarity"
        },
        {
            "doc_id": "doc2",
            "chunk_id": "doc2#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.7,
            "text": "Lower similarity"
        },
        {
            "doc_id": "doc3",
            "chunk_id": "doc3#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.5,
            "text": "Even lower similarity"
        }
    ]
    
    quality = compute_retrieval_quality(chunks)
    
    # Gap should be top1 - top2 = 0.9 - 0.7 = 0.2
    assert quality.confidence.gap is not None
    assert abs(quality.confidence.gap - 0.2) < 0.0001  # Allow small floating point differences
    assert quality.confidence.max == 0.9
    assert abs(quality.confidence.mean - (0.9 + 0.7 + 0.5) / 3) < 0.0001
    assert quality.confidence.hit_count == 3
    
    # Test with single chunk (gap should be None)
    single_chunk = [chunks[0]]
    quality_single = compute_retrieval_quality(single_chunk)
    assert quality_single.confidence.gap is None
    assert quality_single.confidence.max == 0.9
    assert quality_single.confidence.hit_count == 1


def test_empty_chunks():
    """Test retrieval quality with empty chunks list."""
    quality = compute_retrieval_quality([])
    
    assert quality.confidence.max == 0.0
    assert quality.confidence.mean == 0.0
    assert quality.confidence.gap is None
    assert quality.confidence.hit_count == 0
    assert quality.freshness.freshness_violation_count == 0
    assert quality.freshness.freshness_violation is False
    assert quality.top_doc_ids == []
    assert quality.top_timestamps == []


def test_timestamps_extraction():
    """Test that oldest and newest timestamps are correctly identified."""
    chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2021-01-01T00:00:00Z",
            "similarity": 0.8,
            "text": "Oldest"
        },
        {
            "doc_id": "doc2",
            "chunk_id": "doc2#p0",
            "timestamp": "2024-03-15T10:00:00Z",
            "similarity": 0.9,
            "text": "Newest"
        },
        {
            "doc_id": "doc3",
            "chunk_id": "doc3#p0",
            "timestamp": "2023-06-01T00:00:00Z",
            "similarity": 0.7,
            "text": "Middle"
        }
    ]
    
    quality = compute_retrieval_quality(chunks)
    
    # Timestamps are compared as strings, so lexicographic comparison works for ISO format
    assert quality.freshness.oldest_timestamp == "2021-01-01T00:00:00Z"
    assert quality.freshness.newest_timestamp == "2024-03-15T10:00:00Z"
    assert len(quality.top_doc_ids) == 3
    assert len(quality.top_timestamps) == 3

