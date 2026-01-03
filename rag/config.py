"""
Configuration constants for RAG system.
Supports overriding defaults from a config dict.
"""
import json
from pathlib import Path
from typing import Dict, Optional


class Config:
    """Configuration object that holds all configurable parameters."""
    
    def __init__(self, overrides: Optional[Dict] = None):
        """
        Initialize config with optional overrides.
        
        Args:
            overrides: Dict with keys matching config parameters (e.g., top_k, confidence_threshold)
        """
        overrides = overrides or {}
        
        # Retrieval parameters
        self.top_k = overrides.get("top_k", DEFAULT_TOP_K)
        
        # Freshness thresholds
        self.freshness_days = overrides.get("freshness_days", DEFAULT_FRESHNESS_DAYS)
        self.freshness_days_high_risk = overrides.get("freshness_days_high_risk", DEFAULT_FRESHNESS_DAYS_HIGH_RISK)
        
        # Confidence thresholds
        self.confidence_threshold = overrides.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
        self.confidence_threshold_high_risk = overrides.get("confidence_threshold_high_risk", DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK)
        
        # Minimum chunks
        self.min_chunks = overrides.get("min_chunks", DEFAULT_MIN_CHUNKS)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """
        Load config from JSON file.
        
        Args:
            path: Path to JSON config file
        
        Returns:
            Config instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(overrides=config_dict)
    
    def get_freshness_days(self, risk_level: str) -> int:
        """Get freshness threshold for given risk level."""
        if risk_level == "high":
            return self.freshness_days_high_risk
        return self.freshness_days
    
    def get_confidence_threshold(self, risk_level: str) -> float:
        """Get confidence threshold for given risk level."""
        if risk_level == "high":
            return self.confidence_threshold_high_risk
        return self.confidence_threshold


# Default retrieval parameters
DEFAULT_TOP_K = 5

# Default freshness threshold (in days)
DEFAULT_FRESHNESS_DAYS = 90

# Decision engine thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.60
DEFAULT_CONFIDENCE_THRESHOLD_HIGH_RISK = 0.70
DEFAULT_FRESHNESS_DAYS_HIGH_RISK = 30
DEFAULT_MIN_CHUNKS = 2

# Global default config (for backward compatibility)
_default_config = Config()


def get_default_config() -> Config:
    """Get the default config instance."""
    return _default_config
