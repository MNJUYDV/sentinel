"""
Unit tests for citation validation.
"""
import pytest

from rag.validate import validate_citations


def test_blocks_invalid_citation_id():
    """Test that validation blocks when citations contain an ID not in retrieved set."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "Test content"
        },
        {
            "doc_id": "doc2",
            "chunk_id": "doc2#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.8,
            "text": "More content"
        }
    ]
    
    # Valid citation
    valid, errors, warnings = validate_citations(
        ["doc1#p0"],
        retrieved_chunks,
        "Answer text"
    )
    assert valid is True
    assert len(errors) == 0
    
    # Invalid citation (not in retrieved set)
    valid, errors, warnings = validate_citations(
        ["doc3#p0"],  # Not in retrieved chunks
        retrieved_chunks,
        "Answer text"
    )
    assert valid is False
    assert len(errors) > 0
    assert any("does not match" in err.lower() for err in errors)


def test_blocks_empty_citations_non_insufficient_answer():
    """Test that validation blocks when citations are empty but answer is not insufficiency message."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "Test content"
        }
    ]
    
    # Empty citations with non-insufficient answer
    valid, errors, warnings = validate_citations(
        [],
        retrieved_chunks,
        "This is a definitive answer about something."
    )
    assert valid is False
    assert len(errors) > 0
    assert any("empty" in err.lower() and "insufficient" in err.lower() for err in errors)
    
    # Empty citations with insufficient answer (should pass)
    valid, errors, warnings = validate_citations(
        [],
        retrieved_chunks,
        "I don't have enough information to answer this question."
    )
    assert valid is True
    assert len(errors) == 0


def test_passes_valid_citations():
    """Test that validation passes when citations match retrieved chunk IDs."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "First chunk content"
        },
        {
            "doc_id": "doc2",
            "chunk_id": "doc2#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.8,
            "text": "Second chunk content"
        },
        {
            "doc_id": "doc3",
            "chunk_id": "doc3#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.7,
            "text": "Third chunk content"
        }
    ]
    
    # Valid single citation
    valid, errors, warnings = validate_citations(
        ["doc1#p0"],
        retrieved_chunks,
        "Answer text"
    )
    assert valid is True
    assert len(errors) == 0
    
    # Valid multiple citations
    valid, errors, warnings = validate_citations(
        ["doc1#p0", "doc2#p0"],
        retrieved_chunks,
        "Answer text"
    )
    assert valid is True
    assert len(errors) == 0
    
    # Valid all citations
    valid, errors, warnings = validate_citations(
        ["doc1#p0", "doc2#p0", "doc3#p0"],
        retrieved_chunks,
        "Answer text"
    )
    assert valid is True
    assert len(errors) == 0


def test_warning_for_keyword_overlap():
    """Test that warning triggers when citation overlap heuristic finds no overlap (but does NOT block)."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "The policy requires encryption for all sensitive data storage systems."
        }
    ]
    
    # Answer with keywords that overlap (should not warn)
    valid, errors, warnings = validate_citations(
        ["doc1#p0"],
        retrieved_chunks,
        "According to the policy, encryption is required for sensitive data storage."
    )
    assert valid is True
    assert len(errors) == 0
    # Should have no warnings or minimal warnings (keywords overlap)
    keyword_warnings = [w for w in warnings if "keyword overlap" in w.lower() or "not directly support" in w.lower()]
    assert len(keyword_warnings) == 0
    
    # Answer with completely unrelated keywords (should warn but not block)
    valid, errors, warnings = validate_citations(
        ["doc1#p0"],
        retrieved_chunks,
        "The weather forecast predicts sunny skies tomorrow with temperatures in the seventies."
    )
    assert valid is True  # Should still pass validation
    assert len(errors) == 0
    # Should have a warning about keyword overlap
    keyword_warnings = [w for w in warnings if "keyword overlap" in w.lower() or "not directly support" in w.lower()]
    assert len(keyword_warnings) > 0


def test_blocks_duplicate_citations():
    """Test that validation blocks duplicate citations."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "Test content"
        }
    ]
    
    # Duplicate citations
    valid, errors, warnings = validate_citations(
        ["doc1#p0", "doc1#p0"],
        retrieved_chunks,
        "Answer text"
    )
    assert valid is False
    assert len(errors) > 0
    assert any("duplicate" in err.lower() for err in errors)


def test_blocks_too_many_citations():
    """Test that validation blocks when more than MAX_CITATIONS are provided."""
    retrieved_chunks = [
        {
            "doc_id": f"doc{i}",
            "chunk_id": f"doc{i}#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9 - i * 0.1,
            "text": f"Content {i}"
        }
        for i in range(10)  # Create 10 chunks
    ]
    
    # More than 5 citations (MAX_CITATIONS)
    valid, errors, warnings = validate_citations(
        [f"doc{i}#p0" for i in range(6)],  # 6 citations
        retrieved_chunks,
        "Answer text"
    )
    assert valid is False
    assert len(errors) > 0
    assert any("too many" in err.lower() or "maximum" in err.lower() for err in errors)
    
    # Exactly 5 citations (should pass)
    valid, errors, warnings = validate_citations(
        [f"doc{i}#p0" for i in range(5)],  # 5 citations
        retrieved_chunks,
        "Answer text"
    )
    assert valid is True
    assert len(errors) == 0


def test_insufficiency_message_detection():
    """Test that insufficiency message detection works correctly."""
    retrieved_chunks = [
        {
            "doc_id": "doc1",
            "chunk_id": "doc1#p0",
            "timestamp": "2024-01-01T00:00:00Z",
            "similarity": 0.9,
            "text": "Test content"
        }
    ]
    
    # Various insufficiency messages
    insufficiency_phrases = [
        "I don't have enough information",
        "Insufficient context to answer",
        "Cannot provide an answer",
        "Not sure about this",
        "Unable to determine",
        "Missing context",
        "No information available"
    ]
    
    for phrase in insufficiency_phrases:
        valid, errors, warnings = validate_citations(
            [],
            retrieved_chunks,
            phrase
        )
        assert valid is True, f"Should accept insufficiency message: {phrase}"
        assert len(errors) == 0

