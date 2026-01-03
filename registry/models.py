"""
Pydantic models for config version metadata.
"""
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Prompt configuration."""
    system_prompt: str = Field(..., description="System prompt template")
    user_template: str = Field(..., description="User prompt template")


class ConfigHashes(BaseModel):
    """Hashes for config and prompt."""
    config_hash: str = Field(..., description="SHA256 hash of config JSON")
    prompt_hash: str = Field(..., description="SHA256 hash of prompt JSON")


class ConfigVersion(BaseModel):
    """Versioned configuration artifact."""
    config_id: str = Field(..., description="Unique config version ID (e.g., rag_prod_v17)")
    parent_id: Optional[str] = Field(None, description="Parent config ID (for tracking lineage)")
    created_at: str = Field(..., description="ISO datetime when config was created")
    author: str = Field(..., description="Author/creator of this config version")
    change_reason: str = Field(..., description="Reason for this config change")
    config: Dict = Field(..., description="Config dict (top_k, thresholds, etc.)")
    prompt: PromptConfig = Field(..., description="Prompt configuration")
    hashes: ConfigHashes = Field(..., description="Hashes for config and prompt")
    status: str = Field(..., description="Status: dev, staging, or prod")


class ConfigVersionSummary(BaseModel):
    """Summary of config version (for listing)."""
    config_id: str
    parent_id: Optional[str]
    created_at: str
    author: str
    change_reason: str
    status: str


class Pointers(BaseModel):
    """Active config pointers for different environments."""
    dev: str = Field(..., description="Active dev config ID")
    staging: str = Field(..., description="Active staging config ID")
    prod: str = Field(..., description="Active prod config ID")
    prod_previous: Optional[str] = Field(None, description="Previous prod config ID (for rollback)")


class HistoryEvent(BaseModel):
    """Promotion/rollback history event."""
    ts: str = Field(..., description="ISO datetime of event")
    event: str = Field(..., description="Event type: PROMOTE or ROLLBACK")
    from_config: str = Field(..., alias="from", description="Source config ID")
    to_config: str = Field(..., alias="to", description="Target config ID")
    by: str = Field(..., description="Actor who performed the action")
    reason: Optional[str] = Field(None, description="Reason for the action")
    
    class Config:
        allow_population_by_field_name = True

