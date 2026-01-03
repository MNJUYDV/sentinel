#!/usr/bin/env python3
"""
Eval runner CLI for offline evaluation suite.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import Config
from rag.ingest import initialize_rag_index
from rag.risk import classify_risk
from eval.gate import check_gate_and_exit, evaluate_gate, load_gate_rules
from eval.pipeline import run_pipeline
from eval.scoring import score_case

logging.basicConfig(level=logging.WARNING)  # Reduce noise during eval
logger = logging.getLogger(__name__)


def load_golden_set(suite_path: str) -> List[Dict]:
    """Load golden set from JSON file."""
    with open(suite_path, 'r') as f:
        return json.load(f)


def load_config(config_path: str) -> Config:
    """Load config from JSON file and create Config object."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return Config(overrides=config_dict)


def compute_metrics(results: List[Dict], cases: List[Dict]) -> Dict:
    """
    Compute aggregate metrics from results.
    
    Args:
        results: List of result dicts (one per case)
        cases: List of case dicts
    
    Returns:
        Dict with metrics
    """
    total = len(results)
    if total == 0:
        return {}
    
    passed = sum(1 for r in results if r.get("passed", False))
    overall_pass_rate = passed / total if total > 0 else 0.0
    
    # False accept rate: answered when expected abstain/block/clarify
    false_accepts = 0
    false_accept_total = 0
    for result, case in zip(results, cases):
        expected = case["expected_outcome"]
        if expected in ("ABSTAIN", "BLOCK", "CLARIFY"):
            false_accept_total += 1
            if result["decision"] == "ANSWER":
                false_accepts += 1
    false_accept_rate = false_accepts / false_accept_total if false_accept_total > 0 else 0.0
    
    # False refuse rate: abstain/block/clarify when expected answer
    false_refuses = 0
    false_refuse_total = 0
    for result, case in zip(results, cases):
        expected = case["expected_outcome"]
        if expected == "ANSWER":
            false_refuse_total += 1
            if result["decision"] in ("ABSTAIN", "BLOCK", "CLARIFY"):
                false_refuses += 1
    false_refuse_rate = false_refuses / false_refuse_total if false_refuse_total > 0 else 0.0
    
    # Citation validity rate
    citation_valid = sum(
        1 for r in results
        if r.get("attribute_scores", {}).get("citation_validity", {}).get("passed", False)
    )
    citation_validity_rate = citation_valid / total if total > 0 else 0.0
    
    # Staleness correct rate
    staleness_correct = sum(
        1 for r in results
        if r.get("attribute_scores", {}).get("staleness_handling", {}).get("passed", False)
    )
    staleness_cases = sum(
        1 for case in cases
        if case.get("required_behavior", {}).get("must_detect_staleness", False)
    )
    staleness_correct_rate = staleness_correct / staleness_cases if staleness_cases > 0 else 1.0
    
    # Conflict correct rate
    conflict_correct = sum(
        1 for r in results
        if r.get("attribute_scores", {}).get("conflict_handling", {}).get("passed", False)
    )
    conflict_cases = sum(
        1 for case in cases
        if case.get("required_behavior", {}).get("must_surface_conflict", False)
    )
    conflict_correct_rate = conflict_correct / conflict_cases if conflict_cases > 0 else 1.0
    
    # Slice metrics
    slice_metrics = {}
    
    # High-risk slice
    high_risk_results = [r for r, c in zip(results, cases) if c.get("risk_level") == "high"]
    if high_risk_results:
        high_risk_passed = sum(1 for r in high_risk_results if r.get("passed", False))
        slice_metrics["high_risk"] = {
            "overall_pass_rate": high_risk_passed / len(high_risk_results),
            "total": len(high_risk_results),
            "passed": high_risk_passed
        }
    
    # Conflict slice
    conflict_cases_list = [
        (r, c) for r, c in zip(results, cases)
        if c.get("required_behavior", {}).get("must_surface_conflict", False)
    ]
    if conflict_cases_list:
        conflict_results = [r for r, c in conflict_cases_list]
        conflict_passed = sum(1 for r in conflict_results if r.get("passed", False))
        slice_metrics["conflict"] = {
            "overall_pass_rate": conflict_passed / len(conflict_results),
            "total": len(conflict_results),
            "passed": conflict_passed
        }
    
    # Stale slice
    stale_cases_list = [
        (r, c) for r, c in zip(results, cases)
        if c.get("required_behavior", {}).get("must_detect_staleness", False)
    ]
    if stale_cases_list:
        stale_results = [r for r, c in stale_cases_list]
        stale_passed = sum(1 for r in stale_results if r.get("passed", False))
        slice_metrics["stale"] = {
            "overall_pass_rate": stale_passed / len(stale_results),
            "total": len(stale_results),
            "passed": stale_passed
        }
    
    return {
        "overall_pass_rate": overall_pass_rate,
        "false_accept_rate": false_accept_rate,
        "false_refuse_rate": false_refuse_rate,
        "citation_validity_rate": citation_validity_rate,
        "staleness_correct_rate": staleness_correct_rate,
        "conflict_correct_rate": conflict_correct_rate,
        "slice_metrics": slice_metrics,
        "total": total,
        "passed": passed
    }


def run_eval_suite(
    cases: List[Dict],
    vector_index,
    config: Config,
    use_stub_llm: bool = True
) -> List[Dict]:
    """
    Run evaluation suite on cases.
    
    Args:
        cases: List of test cases
        vector_index: VectorIndex instance
        config: Config object
        use_stub_llm: If True, force stub mode
    
    Returns:
        List of result dicts (one per case)
    """
    results = []
    
    for case in cases:
        case_id = case["id"]
        query = case["query"]
        risk_level = case.get("risk_level", "low")
        
        try:
            # Run pipeline
            result = run_pipeline(query, vector_index, config, use_stub_llm=use_stub_llm)
            
            # Score case
            score_result = score_case(case, result, config, risk_level)
            
            # Combine pipeline result with scores
            case_result = {
                "case_id": case_id,
                "query": query,
                "decision": result["decision"],
                "expected_outcome": case["expected_outcome"],
                "passed": score_result["passed"],
                "failure_reasons": score_result["failure_reasons"],
                "attribute_scores": score_result["attribute_scores"],
                "reasons": result["reasons"],
                "signals": result["signals"],
                "retrieval_quality": {
                    "confidence_max": result["retrieval_quality"].confidence.max,
                    "hit_count": result["retrieval_quality"].confidence.hit_count,
                    "freshness_violation": result["retrieval_quality"].freshness.freshness_violation
                },
                "conflicts": {
                    "conflict_detected": result["conflicts"].conflict_detected,
                    "conflict_type": result["conflicts"].conflict_type
                },
                "validation": {
                    "citation_valid": result["validation"].citation_valid,
                    "errors": result["validation"].errors
                }
            }
            results.append(case_result)
            
        except Exception as e:
            logger.error(f"Error processing case {case_id}: {e}", exc_info=True)
            results.append({
                "case_id": case_id,
                "query": query,
                "passed": False,
                "failure_reasons": [f"exception: {str(e)}"],
                "error": str(e)
            })
    
    return results


def print_regression_summary(baseline_metrics: Dict, candidate_metrics: Dict):
    """Print human-readable regression comparison."""
    print("\n" + "=" * 60)
    print("REGRESSION COMPARISON")
    print("=" * 60)
    
    metrics_to_compare = [
        "overall_pass_rate",
        "false_accept_rate",
        "false_refuse_rate",
        "citation_validity_rate",
        "staleness_correct_rate",
        "conflict_correct_rate"
    ]
    
    print(f"\n{'Metric':<30} {'Baseline':<12} {'Candidate':<12} {'Delta':<12}")
    print("-" * 66)
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric, 0.0)
        candidate_val = candidate_metrics.get(metric, 0.0)
        delta = candidate_val - baseline_val
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        
        print(f"{metric:<30} {baseline_val:<12.4f} {candidate_val:<12.4f} {delta_str:<12}")
    
    # Slice metrics
    print("\nSlice Metrics:")
    baseline_slices = baseline_metrics.get("slice_metrics", {})
    candidate_slices = candidate_metrics.get("slice_metrics", {})
    
    for slice_name in set(baseline_slices.keys()) | set(candidate_slices.keys()):
        baseline_slice = baseline_slices.get(slice_name, {})
        candidate_slice = candidate_slices.get(slice_name, {})
        baseline_pass = baseline_slice.get("overall_pass_rate", 0.0)
        candidate_pass = candidate_slice.get("overall_pass_rate", 0.0)
        delta = candidate_pass - baseline_pass
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        
        print(f"  {slice_name:<28} {baseline_pass:<12.4f} {candidate_pass:<12.4f} {delta_str:<12}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run offline evaluation suite")
    parser.add_argument("--suite", required=True, help="Path to golden set JSON file")
    parser.add_argument("--config", required=True, help="Path to config JSON file")
    parser.add_argument("--candidate", help="Path to candidate config JSON file (for regression comparison)")
    parser.add_argument("--baseline", help="Path to baseline results JSON file (for regression comparison)")
    parser.add_argument("--out", required=True, help="Path to output results JSON file")
    parser.add_argument("--gate", help="Path to gate rules JSON file (if provided, will check gate and exit)")
    parser.add_argument("--no-stub", action="store_true", help="Don't force stub LLM mode")
    
    args = parser.parse_args()
    
    # Force stub mode by default (unless --no-stub is provided)
    use_stub_llm = not args.no_stub
    if use_stub_llm:
        # Ensure OpenAI key is not set for deterministic eval
        os.environ.pop("OPENAI_API_KEY", None)
    
    # Load golden set
    cases = load_golden_set(args.suite)
    print(f"Loaded {len(cases)} test cases from {args.suite}")
    
    # Initialize vector index
    script_dir = Path(__file__).parent.parent
    docs_path = script_dir / "data" / "docs.json"
    vector_index, documents = initialize_rag_index(str(docs_path))
    print(f"Initialized vector index with {len(documents)} documents")
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Run eval
    results = run_eval_suite(cases, vector_index, config, use_stub_llm=use_stub_llm)
    
    # Compute metrics
    metrics = compute_metrics(results, cases)
    
    # Save results
    output_data = {
        "config": args.config,
        "suite": args.suite,
        "metrics": metrics,
        "results": results
    }
    
    with open(args.out, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Saved results to {args.out}")
    
    # Print summary
    print(f"\nMetrics:")
    print(f"  Overall pass rate: {metrics['overall_pass_rate']:.4f} ({metrics['passed']}/{metrics['total']})")
    print(f"  False accept rate: {metrics['false_accept_rate']:.4f}")
    print(f"  False refuse rate: {metrics['false_refuse_rate']:.4f}")
    print(f"  Citation validity rate: {metrics['citation_validity_rate']:.4f}")
    
    # Regression comparison if baseline provided
    if args.baseline:
        with open(args.baseline, 'r') as f:
            baseline_data = json.load(f)
        baseline_metrics = baseline_data.get("metrics", {})
        print_regression_summary(baseline_metrics, metrics)
        
        # If gate rules provided, check gate with baseline comparison
        if args.gate:
            check_gate_and_exit(metrics, args.gate, baseline_metrics)
    elif args.gate:
        # Check gate without baseline comparison
        check_gate_and_exit(metrics, args.gate)
    elif args.candidate:
        # Run candidate config and compare
        candidate_config = load_config(args.candidate)
        print(f"\nRunning candidate config from {args.candidate}")
        candidate_results = run_eval_suite(cases, vector_index, candidate_config, use_stub_llm=use_stub_llm)
        candidate_metrics = compute_metrics(candidate_results, cases)
        
        # Save candidate results
        candidate_out = args.out.replace(".json", "_candidate.json")
        candidate_output_data = {
            "config": args.candidate,
            "suite": args.suite,
            "metrics": candidate_metrics,
            "results": candidate_results
        }
        with open(candidate_out, 'w') as f:
            json.dump(candidate_output_data, f, indent=2, default=str)
        print(f"Saved candidate results to {candidate_out}")
        
        print_regression_summary(metrics, candidate_metrics)
        
        # Check gate if provided
        if args.gate:
            check_gate_and_exit(candidate_metrics, args.gate, metrics)


if __name__ == "__main__":
    main()

