"""
Promotion and rollback workflows for config management.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from registry.history import log_event
from registry.store import get_pointers, get_version, set_pointer

logger = logging.getLogger(__name__)


def _run_eval_gate(candidate_config_id: str, prod_config_id: str, gate_rules_path: str) -> Dict:
    """
    Run eval gate comparing candidate vs prod config.
    
    Args:
        candidate_config_id: Candidate config ID
        prod_config_id: Current prod config ID
        gate_rules_path: Path to gate rules JSON file
    
    Returns:
        Dict with: gate_passed (bool), failures (List[str]), results_path (str)
    """
    # Load config versions
    candidate_version = get_version(candidate_config_id)
    prod_version = get_version(prod_config_id)
    
    if not candidate_version:
        return {"gate_passed": False, "failures": [f"Candidate config {candidate_config_id} not found"]}
    if not prod_version:
        return {"gate_passed": False, "failures": [f"Prod config {prod_config_id} not found"]}
    
    # Write temp config files for eval
    script_dir = Path(__file__).parent.parent
    eval_dir = script_dir / "eval"
    temp_candidate_config = eval_dir / "temp_candidate_config.json"
    temp_prod_config = eval_dir / "temp_prod_config.json"
    
    try:
        # Write config files (only the config dict, not metadata)
        with open(temp_candidate_config, 'w') as f:
            json.dump(candidate_version.config, f, indent=2)
        with open(temp_prod_config, 'w') as f:
            json.dump(prod_version.config, f, indent=2)
        
        # Run eval with baseline (prod) and candidate
        suite_path = eval_dir / "golden_set_v1.json"
        results_path = eval_dir / "last_results.json"
        
        # Ensure OPENAI_API_KEY is not set for deterministic eval
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        
        # Run eval runner with baseline and candidate
        cmd = [
            sys.executable, "-m", "eval.run_eval",
            "--suite", str(suite_path),
            "--config", str(temp_prod_config),
            "--candidate", str(temp_candidate_config),
            "--out", str(results_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=script_dir)
        
        if result.returncode != 0:
            logger.error(f"Eval runner failed: {result.stderr}")
            return {"gate_passed": False, "failures": [f"Eval runner failed: {result.stderr[:200]}"]}
        
        # Load results - eval runner saves baseline to results_path and candidate to candidate_path
        try:
            # Load baseline metrics (from baseline run)
            with open(results_path, 'r') as f:
                baseline_data = json.load(f)
            baseline_metrics = baseline_data.get("metrics", {})
            
            # Load candidate results (saved as _candidate.json by eval runner)
            candidate_results_path = eval_dir / "last_results_candidate.json"
            if not candidate_results_path.exists():
                # Fallback: if candidate file doesn't exist, something went wrong
                return {
                    "gate_passed": False,
                    "failures": ["Candidate results file not found"],
                    "results_path": str(results_path)
                }
            
            with open(candidate_results_path, 'r') as f:
                candidate_data = json.load(f)
            candidate_metrics = candidate_data.get("metrics", {})
            
            # Evaluate gate on candidate metrics with baseline comparison
            from eval.gate import evaluate_gate, load_gate_rules
            
            gate_rules = load_gate_rules(gate_rules_path)
            gate_result = evaluate_gate(candidate_metrics, gate_rules, baseline_metrics)
            
            return {
                "gate_passed": gate_result["passed"],
                "failures": gate_result["failures"],
                "results_path": str(results_path)
            }
        except Exception as e:
            logger.error(f"Failed to evaluate gate: {e}", exc_info=True)
            return {
                "gate_passed": False,
                "failures": [f"Gate evaluation failed: {str(e)}"],
                "results_path": str(results_path)
            }
    
    finally:
        # Cleanup temp files
        for temp_file in [temp_candidate_config, temp_prod_config]:
            if temp_file.exists():
                temp_file.unlink()


def promote_to_prod(candidate_id: str, actor: str, gate_rules_path: Optional[str] = None) -> Dict:
    """
    Promote candidate config to prod if gate passes.
    
    Args:
        candidate_id: Candidate config ID to promote
        actor: Actor performing promotion
        gate_rules_path: Path to gate rules (defaults to eval/gate_rules.json)
    
    Returns:
        Dict with: promoted (bool), gate_passed (bool), gate_failures (List), from (str), to (str)
    """
    # Get current pointers
    pointers = get_pointers()
    current_prod = pointers.prod
    
    # Validate candidate exists
    candidate_version = get_version(candidate_id)
    if not candidate_version:
        return {
            "promoted": False,
            "gate_passed": False,
            "gate_failures": [f"Candidate config {candidate_id} does not exist"],
            "from": current_prod,
            "to": candidate_id
        }
    
    # Use default gate rules path if not provided
    if gate_rules_path is None:
        script_dir = Path(__file__).parent.parent
        gate_rules_path = script_dir / "eval" / "gate_rules.json"
    
    # Run eval gate
    gate_result = _run_eval_gate(candidate_id, current_prod, str(gate_rules_path))
    
    if not gate_result["gate_passed"]:
        return {
            "promoted": False,
            "gate_passed": False,
            "gate_failures": gate_result["failures"],
            "from": current_prod,
            "to": candidate_id
        }
    
    # Gate passed - promote
    # Update pointers: prod_previous = current prod, prod = candidate
    pointers.prod_previous = current_prod
    pointers.prod = candidate_id
    
    # Save pointers
    from registry.store import REGISTRY_BASE, POINTERS_FILE
    with open(POINTERS_FILE, 'w') as f:
        json.dump(pointers.dict(), f, indent=2)
    
    # Log promotion event
    log_event("PROMOTE", current_prod, candidate_id, actor, reason="Gate passed")
    
    logger.info(f"Promoted {candidate_id} to prod (from {current_prod})")
    
    return {
        "promoted": True,
        "gate_passed": True,
        "gate_failures": [],
        "from": current_prod,
        "to": candidate_id
    }


def rollback_prod(actor: str, reason: Optional[str] = None) -> Dict:
    """
    Rollback prod to previous version.
    
    Args:
        actor: Actor performing rollback
        reason: Optional reason for rollback
    
    Returns:
        Dict with: rolled_back (bool), from (str), to (str)
    """
    pointers = get_pointers()
    
    if not pointers.prod_previous:
        return {
            "rolled_back": False,
            "from": pointers.prod,
            "to": None,
            "error": "No previous prod config to rollback to"
        }
    
    current_prod = pointers.prod
    previous_prod = pointers.prod_previous
    
    # Validate previous config exists
    if not get_version(previous_prod):
        return {
            "rolled_back": False,
            "from": current_prod,
            "to": previous_prod,
            "error": f"Previous prod config {previous_prod} does not exist"
        }
    
    # Rollback: prod = prod_previous, prod_previous = None (one-level rollback only)
    pointers.prod = previous_prod
    pointers.prod_previous = None
    
    # Save pointers
    from registry.store import POINTERS_FILE
    with open(POINTERS_FILE, 'w') as f:
        json.dump(pointers.dict(), f, indent=2)
    
    # Log rollback event
    log_event("ROLLBACK", current_prod, previous_prod, actor, reason=reason)
    
    logger.info(f"Rolled back prod from {current_prod} to {previous_prod}")
    
    return {
        "rolled_back": True,
        "from": current_prod,
        "to": previous_prod
    }

