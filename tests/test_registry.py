"""
Tests for config registry and promotion workflows.
"""
import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from registry.models import ConfigVersion, PromptConfig
from registry.store import (
    create_version, get_version, list_versions, get_pointers, set_pointer, get_prod_history
)
from registry.promotion import promote_to_prod, rollback_prod
from registry.history import log_event
from rag.config import Config


@pytest.fixture
def temp_registry(tmp_path):
    """Create a temporary registry directory for testing."""
    registry_base = tmp_path / "configs" / "registry"
    versions_dir = registry_base / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    
    # Mock the REGISTRY_BASE path
    original_base = Path(__file__).parent.parent / "configs" / "registry"
    
    # Create pointers file
    pointers_file = registry_base / "pointers.json"
    pointers_file.write_text(json.dumps({
        "dev": "rag_dev_v1",
        "staging": "rag_staging_v1",
        "prod": "rag_prod_v1",
        "prod_previous": None
    }))
    
    # Patch REGISTRY_BASE in store module
    with patch('registry.store.REGISTRY_BASE', registry_base):
        with patch('registry.store.VERSIONS_DIR', versions_dir):
            with patch('registry.store.POINTERS_FILE', pointers_file):
                with patch('registry.history.HISTORY_FILE', registry_base / "history.jsonl"):
                    yield registry_base


class TestConfigStore:
    """Test config store functions."""
    
    def test_create_version(self, temp_registry):
        """Test creating a new config version."""
        config = {
            "top_k": 5,
            "confidence_threshold": 0.6,
            "confidence_threshold_high_risk": 0.7,
            "freshness_days": 90,
            "freshness_days_high_risk": 30,
            "min_chunks": 2
        }
        prompt = {
            "system_prompt": "You are a helpful assistant.",
            "user_template": "Query: {query}\n\nContext:\n{context}"
        }
        
        version = create_version(
            parent_id=None,
            author="test_user",
            change_reason="Initial config",
            config=config,
            prompt=prompt
        )
        
        assert version.config_id is not None
        assert version.author == "test_user"
        assert version.config == config
        assert version.hashes.config_hash is not None
        assert version.hashes.prompt_hash is not None
    
    def test_get_version(self, temp_registry):
        """Test retrieving a config version."""
        config = {"top_k": 5, "confidence_threshold": 0.6}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        version = create_version(None, "test", "test", config, prompt)
        retrieved = get_version(version.config_id)
        
        assert retrieved is not None
        assert retrieved.config_id == version.config_id
        assert retrieved.config == config
    
    def test_list_versions(self, temp_registry):
        """Test listing config versions."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        v1 = create_version(None, "test", "v1", config, prompt)
        v2 = create_version(v1.config_id, "test", "v2", config, prompt)
        
        versions = list_versions()
        assert len(versions) >= 2
        assert any(v.config_id == v1.config_id for v in versions)
        assert any(v.config_id == v2.config_id for v in versions)
    
    def test_set_pointer(self, temp_registry):
        """Test setting environment pointer."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        version = create_version(None, "test", "test", config, prompt)
        
        set_pointer("dev", version.config_id)
        pointers = get_pointers()
        assert pointers.dev == version.config_id
    
    def test_get_pointer(self, temp_registry):
        """Test getting environment pointer."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        version = create_version(None, "test", "test", config, prompt)
        
        set_pointer("prod", version.config_id)
        pointer = get_pointer("prod")
        assert pointer == version.config_id


class TestPromotion:
    """Test promotion and rollback workflows."""
    
    @patch('registry.promotion._run_eval_gate')
    def test_promote_to_prod_success(self, mock_gate, temp_registry):
        """Test successful promotion when gate passes."""
        config = {"top_k": 5, "confidence_threshold": 0.6}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        prod_version = create_version(None, "test", "prod", config, prompt)
        candidate_version = create_version(prod_version.config_id, "test", "candidate", config, prompt)
        
        set_pointer("prod", prod_version.config_id)
        
        # Mock gate to pass
        mock_gate.return_value = {
            "gate_passed": True,
            "failures": [],
            "results_path": "/tmp/results.json"
        }
        
        result = promote_to_prod(candidate_version.config_id, "test_actor")
        
        assert result["promoted"] is True
        assert result["gate_passed"] is True
        
        # Check pointer updated
        pointers = get_pointers()
        assert pointers.prod == candidate_version.config_id
        assert pointers.prod_previous == prod_version.config_id
    
    @patch('registry.promotion._run_eval_gate')
    def test_promote_to_prod_failure(self, mock_gate, temp_registry):
        """Test promotion failure when gate fails."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        prod_version = create_version(None, "test", "prod", config, prompt)
        candidate_version = create_version(prod_version.config_id, "test", "candidate", config, prompt)
        
        set_pointer("prod", prod_version.config_id)
        
        # Mock gate to fail
        mock_gate.return_value = {
            "gate_passed": False,
            "failures": ["overall_pass_rate 0.5 < min 0.85"],
            "results_path": "/tmp/results.json"
        }
        
        result = promote_to_prod(candidate_version.config_id, "test_actor")
        
        assert result["promoted"] is False
        assert result["gate_passed"] is False
        assert len(result["gate_failures"]) > 0
        
        # Check pointer NOT updated
        pointers = get_pointers()
        assert pointers.prod == prod_version.config_id
    
    def test_rollback_prod(self, temp_registry):
        """Test rolling back prod."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        v1 = create_version(None, "test", "v1", config, prompt)
        v2 = create_version(v1.config_id, "test", "v2", config, prompt)
        
        set_pointer("prod", v2.config_id)
        # Manually set prod_previous for test
        with patch('registry.store.POINTERS_FILE') as mock_file:
            mock_file.parent = temp_registry
            pointers = get_pointers()
            pointers.prod = v2.config_id
            pointers.prod_previous = v1.config_id
            pointers_file = temp_registry / "pointers.json"
            with open(pointers_file, 'w') as f:
                json.dump(pointers.dict(), f)
        
        # Patch pointers file for rollback
        with patch('registry.store.POINTERS_FILE', temp_registry / "pointers.json"):
            with patch('registry.promotion.POINTERS_FILE', temp_registry / "pointers.json"):
                # Load current state
                with open(temp_registry / "pointers.json", 'r') as f:
                    data = json.load(f)
                data["prod"] = v2.config_id
                data["prod_previous"] = v1.config_id
                with open(temp_registry / "pointers.json", 'w') as f:
                    json.dump(data, f)
                
                result = rollback_prod("test_actor", "Test rollback")
                
                assert result["rolled_back"] is True
                assert result["to"] == v1.config_id
                
                # Check pointer updated
                pointers = get_pointers()
                assert pointers.prod == v1.config_id
    
    def test_rollback_no_previous(self, temp_registry):
        """Test rollback fails when no previous prod."""
        config = {"top_k": 5}
        prompt = {"system_prompt": "Test", "user_template": "Query: {query}"}
        
        v1 = create_version(None, "test", "v1", config, prompt)
        set_pointer("prod", v1.config_id)
        
        result = rollback_prod("test_actor", "Test")
        
        assert result["rolled_back"] is False
        assert "error" in result

