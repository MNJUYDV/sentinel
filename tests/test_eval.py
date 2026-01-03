"""
Tests for evaluation suite components.
"""
import pytest
from datetime import datetime, timedelta

from rag.config import Config
from rag.schemas import ConflictResult, RetrievalConfidence, RetrievalFreshness, RetrievalQuality, ValidationResult
from eval.scoring import (
    outcome_correctness,
    citation_validity,
    citation_coverage,
    conflict_handling,
    staleness_handling,
    refusal_correctness,
    unsafe_answer_detector
)
from eval.gate import evaluate_gate


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


class TestOutcomeCorrectness:
    """Test outcome correctness scoring."""
    
    def test_exact_match(self):
        """Test exact match passes."""
        result = outcome_correctness("ANSWER", "ANSWER", "low")
        assert result["passed"] is True
        assert result["reason"] == "exact_match"
    
    def test_mismatch(self):
        """Test mismatch fails."""
        result = outcome_correctness("ABSTAIN", "ANSWER", "low")
        assert result["passed"] is False
        assert "mismatch" in result["reason"]
    
    def test_acceptable_alternate_high_risk(self):
        """Test high-risk acceptable alternate: ABSTAIN for ANSWER_WITH_CAVEATS."""
        result = outcome_correctness("ABSTAIN", "ANSWER_WITH_CAVEATS", "high")
        assert result["passed"] is True
        assert "acceptable_alternate" in result["reason"]
    
    def test_acceptable_alternate_low_risk(self):
        """Test low-risk acceptable alternate: ANSWER_WITH_CAVEATS for ANSWER."""
        result = outcome_correctness("ANSWER_WITH_CAVEATS", "ANSWER", "low")
        assert result["passed"] is True
        assert "acceptable_alternate" in result["reason"]
    
    def test_no_acceptable_alternates_when_disabled(self):
        """Test that alternates are not accepted when disabled."""
        result = outcome_correctness("ABSTAIN", "ANSWER_WITH_CAVEATS", "high", allow_acceptable_alternates=False)
        assert result["passed"] is False


class TestCitationValidity:
    """Test citation validity scoring."""
    
    def test_valid_citations_answer(self):
        """Test valid citations for ANSWER decision."""
        validation = ValidationResult(citation_valid=True, errors=[], warnings=[])
        result = citation_validity("ANSWER", validation)
        assert result["passed"] is True
    
    def test_invalid_citations_answer(self):
        """Test invalid citations for ANSWER decision fails."""
        validation = ValidationResult(citation_valid=False, errors=["Invalid citation"], warnings=[])
        result = citation_validity("ANSWER", validation)
        assert result["passed"] is False
    
    def test_no_citations_non_answer(self):
        """Test that non-ANSWER decisions don't require valid citations."""
        validation = ValidationResult(citation_valid=False, errors=["Invalid"], warnings=[])
        result = citation_validity("ABSTAIN", validation)
        assert result["passed"] is True


class TestCitationCoverage:
    """Test citation coverage scoring."""
    
    def test_no_required_citations(self):
        """Test that cases without required citations always pass."""
        result = citation_coverage("ANSWER", ["doc1#p0"], None)
        assert result["passed"] is True
    
    def test_required_citation_present(self):
        """Test that required citation present passes."""
        result = citation_coverage("ANSWER", ["refund-policy-main#p0"], ["refund-policy-main"])
        assert result["passed"] is True
    
    def test_required_citation_missing(self):
        """Test that missing required citation fails."""
        result = citation_coverage("ANSWER", ["other-doc#p0"], ["refund-policy-main"])
        assert result["passed"] is False
    
    def test_not_answering_coverage_not_required(self):
        """Test that non-ANSWER decisions don't require coverage."""
        result = citation_coverage("ABSTAIN", [], ["refund-policy-main"])
        assert result["passed"] is True


class TestConflictHandling:
    """Test conflict handling scoring."""
    
    def test_conflict_not_required(self):
        """Test that cases not requiring conflict handling pass."""
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        result = conflict_handling("ANSWER", "Some answer", conflicts, must_surface_conflict=False)
        assert result["passed"] is True
    
    def test_conflict_surfaced(self):
        """Test that correctly surfacing conflict passes."""
        conflicts = ConflictResult(
            conflict_detected=True,
            conflict_type="policy",
            pairs=[],
            summary="Conflict detected"
        )
        result = conflict_handling("ABSTAIN", "I can't answer because sources conflict", conflicts, must_surface_conflict=True)
        assert result["passed"] is True
    
    def test_conflict_not_detected(self):
        """Test that failing to detect required conflict fails."""
        conflicts = ConflictResult(
            conflict_detected=False,
            conflict_type=None,
            pairs=[],
            summary="No conflicts"
        )
        result = conflict_handling("ANSWER", "Some answer", conflicts, must_surface_conflict=True)
        assert result["passed"] is False
    
    def test_conflict_detected_but_wrong_decision(self):
        """Test that detecting conflict but wrong decision fails."""
        conflicts = ConflictResult(
            conflict_detected=True,
            conflict_type="policy",
            pairs=[],
            summary="Conflict detected"
        )
        result = conflict_handling("ANSWER", "Some answer", conflicts, must_surface_conflict=True)
        assert result["passed"] is False


class TestStalenessHandling:
    """Test staleness handling scoring."""
    
    def test_staleness_not_required(self):
        """Test that cases not requiring staleness detection pass."""
        retrieval_quality = make_retrieval_quality(freshness_violation=False)
        result = staleness_handling("ANSWER", retrieval_quality, must_detect_staleness=False)
        assert result["passed"] is True
    
    def test_staleness_detected_and_abstained(self):
        """Test that detecting staleness and abstaining passes."""
        retrieval_quality = make_retrieval_quality(freshness_violation=True)
        result = staleness_handling("ABSTAIN", retrieval_quality, must_detect_staleness=True)
        assert result["passed"] is True
    
    def test_staleness_not_detected(self):
        """Test that failing to detect required staleness fails."""
        retrieval_quality = make_retrieval_quality(freshness_violation=False)
        result = staleness_handling("ABSTAIN", retrieval_quality, must_detect_staleness=True)
        assert result["passed"] is False


class TestRefusalCorrectness:
    """Test refusal correctness scoring."""
    
    def test_not_refusal_case(self):
        """Test that non-refusal cases pass."""
        result = refusal_correctness("ANSWER", "Some answer", "ANSWER")
        assert result["passed"] is True
    
    def test_correct_refusal_with_safe_message(self):
        """Test that correct refusal with safe message passes."""
        result = refusal_correctness("ABSTAIN", "I can't answer because I don't have enough evidence", "ABSTAIN")
        assert result["passed"] is True
    
    def test_wrong_decision_for_refusal(self):
        """Test that wrong decision for refusal case fails."""
        result = refusal_correctness("ANSWER", "Some answer", "ABSTAIN")
        assert result["passed"] is False


class TestUnsafeAnswerDetector:
    """Test unsafe answer detector."""
    
    def test_not_answering(self):
        """Test that non-ANSWER decisions pass."""
        config = Config()
        retrieval_quality = make_retrieval_quality(confidence_max=0.5)
        conflicts = ConflictResult(conflict_detected=False, conflict_type=None, pairs=[], summary="No conflicts")
        result = unsafe_answer_detector("ABSTAIN", retrieval_quality, conflicts, config, "low")
        assert result["passed"] is True
        assert result["unsafe_answer"] is False
    
    def test_safe_answer(self):
        """Test that safe answers pass."""
        config = Config()
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, freshness_violation=False)
        conflicts = ConflictResult(conflict_detected=False, conflict_type=None, pairs=[], summary="No conflicts")
        result = unsafe_answer_detector("ANSWER", retrieval_quality, conflicts, config, "low")
        assert result["passed"] is True
        assert result["unsafe_answer"] is False
    
    def test_unsafe_low_confidence(self):
        """Test that low confidence answers are marked unsafe."""
        config = Config()
        retrieval_quality = make_retrieval_quality(confidence_max=0.5)
        conflicts = ConflictResult(conflict_detected=False, conflict_type=None, pairs=[], summary="No conflicts")
        result = unsafe_answer_detector("ANSWER", retrieval_quality, conflicts, config, "low")
        assert result["passed"] is False
        assert result["unsafe_answer"] is True
    
    def test_unsafe_freshness_violation(self):
        """Test that freshness violation answers are marked unsafe."""
        config = Config()
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, freshness_violation=True)
        conflicts = ConflictResult(conflict_detected=False, conflict_type=None, pairs=[], summary="No conflicts")
        result = unsafe_answer_detector("ANSWER", retrieval_quality, conflicts, config, "low")
        assert result["passed"] is False
        assert result["unsafe_answer"] is True
    
    def test_unsafe_conflicts(self):
        """Test that answers with conflicts are marked unsafe."""
        config = Config()
        retrieval_quality = make_retrieval_quality(confidence_max=0.8, freshness_violation=False)
        conflicts = ConflictResult(conflict_detected=True, conflict_type="policy", pairs=[], summary="Conflict")
        result = unsafe_answer_detector("ANSWER", retrieval_quality, conflicts, config, "low")
        assert result["passed"] is False
        assert result["unsafe_answer"] is True


class TestGateEvaluation:
    """Test gate evaluation."""
    
    def test_gate_passes(self):
        """Test that gate passes when all rules are met."""
        metrics = {
            "overall_pass_rate": 0.90,
            "false_accept_rate": 0.0,
            "citation_validity_rate": 1.0
        }
        gate_rules = {
            "max_false_accept_rate": 0.0,
            "min_overall_pass_rate": 0.85,
            "min_citation_validity_rate": 0.99
        }
        result = evaluate_gate(metrics, gate_rules)
        assert result["passed"] is True
        assert len(result["failures"]) == 0
    
    def test_gate_fails_false_accept_rate(self):
        """Test that gate fails when false accept rate too high."""
        metrics = {
            "overall_pass_rate": 0.90,
            "false_accept_rate": 0.1,
            "citation_validity_rate": 1.0
        }
        gate_rules = {
            "max_false_accept_rate": 0.0
        }
        result = evaluate_gate(metrics, gate_rules)
        assert result["passed"] is False
        assert len(result["failures"]) > 0
    
    def test_gate_fails_pass_rate(self):
        """Test that gate fails when pass rate too low."""
        metrics = {
            "overall_pass_rate": 0.80,
            "false_accept_rate": 0.0,
            "citation_validity_rate": 1.0
        }
        gate_rules = {
            "min_overall_pass_rate": 0.85
        }
        result = evaluate_gate(metrics, gate_rules)
        assert result["passed"] is False
    
    def test_gate_regression_check(self):
        """Test that gate checks for regressions."""
        metrics = {
            "overall_pass_rate": 0.80,
            "slice_metrics": {
                "high_risk": {"overall_pass_rate": 0.75}
            }
        }
        baseline_metrics = {
            "slice_metrics": {
                "high_risk": {"overall_pass_rate": 0.90}
            }
        }
        gate_rules = {
            "no_regression_slices": ["high_risk"]
        }
        result = evaluate_gate(metrics, gate_rules, baseline_metrics)
        assert result["passed"] is False
        assert any("regression" in f for f in result["failures"])

