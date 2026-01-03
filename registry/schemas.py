"""
Request/response schemas for config registry API endpoints.
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from registry.models import ConfigVersion, ConfigVersionSummary, Pointers


class CreateConfigRequest(BaseModel):
    """Request to create a new config version."""
    parent_id: Optional[str] = Field(None, description="Parent config ID")
    author: str = Field(..., description="Author name")
    change_reason: str = Field(..., description="Reason for change")
    config: Dict = Field(..., description="Config dict (top_k, thresholds, etc.)")
    prompt: Dict = Field(..., description="Prompt dict with system_prompt and user_template")


class PromoteRequest(BaseModel):
    """Request to promote candidate to prod."""
    candidate_id: str = Field(..., description="Candidate config ID to promote")
    actor: str = Field(..., description="Actor performing promotion")


class RollbackRequest(BaseModel):
    """Request to rollback prod."""
    actor: str = Field(..., description="Actor performing rollback")
    reason: Optional[str] = Field(None, description="Reason for rollback")


class PromoteResponse(BaseModel):
    """Response from promote endpoint."""
    promoted: bool = Field(..., description="Whether promotion succeeded")
    gate_passed: bool = Field(..., description="Whether gate passed")
    gate_failures: List[str] = Field(default_factory=list, description="Gate failure reasons")
    from_config: str = Field(..., alias="from", description="Source config ID")
    to_config: str = Field(..., alias="to", description="Target config ID")
    
    class Config:
        allow_population_by_field_name = True


class RollbackResponse(BaseModel):
    """Response from rollback endpoint."""
    rolled_back: bool = Field(..., description="Whether rollback succeeded")
    from_config: str = Field(..., alias="from", description="Source config ID")
    to_config: Optional[str] = Field(None, alias="to", description="Target config ID")
    error: Optional[str] = Field(None, description="Error message if rollback failed")
    
    class Config:
        allow_population_by_field_name = True

