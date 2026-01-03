"""
Gate evaluation for blocking rollouts when behavior regresses.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional


def load_gate_rules(gate_rules_path: str) -> Dict:
    """Load gate rules from JSON file."""
    with open(gate_rules_path, 'r') as f:
        return json.load(f)


def evaluate_gate(
    metrics: Dict,
    gate_rules: Dict,
    baseline_metrics: Optional[Dict] = None
) -> Dict:
    """
    Evaluate gate rules against metrics.
    
    Args:
        metrics: Metrics dict with keys like overall_pass_rate, false_accept_rate, etc.
        gate_rules: Gate rules dict
        baseline_metrics: Optional baseline metrics for regression checks
    
    Returns:
        Dict with: passed (bool), failures (List[str])
    """
    failures = []
    
    # Check max_false_accept_rate
    if "max_false_accept_rate" in gate_rules:
        max_far = gate_rules["max_false_accept_rate"]
        if metrics.get("false_accept_rate", 0.0) > max_far:
            failures.append(
                f"false_accept_rate {metrics['false_accept_rate']:.4f} > max {max_far}"
            )
    
    # Check min_overall_pass_rate
    if "min_overall_pass_rate" in gate_rules:
        min_pass = gate_rules["min_overall_pass_rate"]
        if metrics.get("overall_pass_rate", 0.0) < min_pass:
            failures.append(
                f"overall_pass_rate {metrics['overall_pass_rate']:.4f} < min {min_pass}"
            )
    
    # Check min_citation_validity_rate
    if "min_citation_validity_rate" in gate_rules:
        min_citation = gate_rules["min_citation_validity_rate"]
        if metrics.get("citation_validity_rate", 0.0) < min_citation:
            failures.append(
                f"citation_validity_rate {metrics['citation_validity_rate']:.4f} < min {min_citation}"
            )
    
    # Check no_regression_slices
    if baseline_metrics and "no_regression_slices" in gate_rules:
        no_regression_slices = gate_rules["no_regression_slices"]
        slice_metrics = metrics.get("slice_metrics", {})
        baseline_slice_metrics = baseline_metrics.get("slice_metrics", {})
        
        for slice_name in no_regression_slices:
            if slice_name in slice_metrics and slice_name in baseline_slice_metrics:
                candidate_pass = slice_metrics[slice_name].get("overall_pass_rate", 0.0)
                baseline_pass = baseline_slice_metrics[slice_name].get("overall_pass_rate", 0.0)
                
                if candidate_pass < baseline_pass:
                    failures.append(
                        f"regression in {slice_name}: {candidate_pass:.4f} < baseline {baseline_pass:.4f}"
                    )
    
    passed = len(failures) == 0
    
    return {
        "passed": passed,
        "failures": failures
    }


def check_gate_and_exit(
    metrics: Dict,
    gate_rules_path: str,
    baseline_metrics: Optional[Dict] = None
):
    """
    Check gate rules and exit with appropriate code.
    
    Args:
        metrics: Candidate metrics
        gate_rules_path: Path to gate rules JSON file
        baseline_metrics: Optional baseline metrics
    
    Exits:
        sys.exit(0) if gate passes
        sys.exit(1) if gate fails
    """
    gate_rules = load_gate_rules(gate_rules_path)
    gate_result = evaluate_gate(metrics, gate_rules, baseline_metrics)
    
    if not gate_result["passed"]:
        print("GATE FAILED")
        print("\nGate rule violations:")
        for failure in gate_result["failures"]:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print("GATE PASSED")
        sys.exit(0)

