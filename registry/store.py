"""
Registry store for managing config versions and pointers.
File-based storage for MVP.
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from registry.models import ConfigVersion, ConfigHashes, Pointers, ConfigVersionSummary, PromptConfig

logger = logging.getLogger(__name__)

# Registry base directory
REGISTRY_BASE = Path(__file__).parent.parent / "configs" / "registry"
VERSIONS_DIR = REGISTRY_BASE / "versions"
POINTERS_FILE = REGISTRY_BASE / "pointers.json"
HISTORY_FILE = REGISTRY_BASE / "history.jsonl"


def _ensure_registry_dirs():
    """Ensure registry directories exist."""
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_BASE.mkdir(parents=True, exist_ok=True)


def _compute_hash(data: Dict) -> str:
    """Compute SHA256 hash of normalized JSON."""
    # Normalize by sorting keys and using compact JSON
    normalized = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def _get_version_file(config_id: str) -> Path:
    """Get path to version file."""
    return VERSIONS_DIR / f"{config_id}.json"


def list_versions() -> List[ConfigVersionSummary]:
    """List all config versions (summary only)."""
    _ensure_registry_dirs()
    
    versions = []
    for version_file in VERSIONS_DIR.glob("*.json"):
        try:
            with open(version_file, 'r') as f:
                data = json.load(f)
            versions.append(ConfigVersionSummary(**data))
        except Exception as e:
            logger.warning(f"Failed to load version {version_file}: {e}")
    
    # Sort by created_at descending (newest first)
    versions.sort(key=lambda v: v.created_at, reverse=True)
    return versions


def get_version(config_id: str) -> Optional[ConfigVersion]:
    """Get full config version by ID."""
    _ensure_registry_dirs()
    
    version_file = _get_version_file(config_id)
    if not version_file.exists():
        return None
    
    try:
        with open(version_file, 'r') as f:
            data = json.load(f)
        return ConfigVersion(**data)
    except Exception as e:
        logger.error(f"Failed to load version {config_id}: {e}")
        return None


def create_version(
    parent_id: Optional[str],
    author: str,
    change_reason: str,
    config: Dict,
    prompt: Dict,
    config_id: Optional[str] = None
) -> ConfigVersion:
    """
    Create a new config version.
    
    Args:
        parent_id: Parent config ID (optional)
        author: Author name
        change_reason: Reason for change
        config: Config dict (for rag/config.py)
        prompt: Prompt dict with system_prompt and user_template
        config_id: Optional explicit config_id (auto-generated if None)
    
    Returns:
        Created ConfigVersion
    """
    _ensure_registry_dirs()
    
    # Generate config_id if not provided
    if config_id is None:
        # Simple auto-increment based on existing versions
        existing = list_versions()
        if existing:
            # Extract version numbers and find max
            max_num = 0
            for v in existing:
                if v.config_id.startswith("rag_"):
                    parts = v.config_id.split("_")
                    if len(parts) >= 3:
                        try:
                            num = int(parts[-1].replace("v", ""))
                            max_num = max(max_num, num)
                        except ValueError:
                            pass
            config_id = f"rag_config_v{max_num + 1}"
        else:
            config_id = "rag_config_v1"
    
    # Validate parent exists if provided
    if parent_id and get_version(parent_id) is None:
        raise ValueError(f"Parent config {parent_id} does not exist")
    
    # Create prompt config
    prompt_config = PromptConfig(**prompt)
    
    # Compute hashes
    config_hash = _compute_hash(config)
    prompt_hash = _compute_hash(prompt)
    hashes = ConfigHashes(config_hash=config_hash, prompt_hash=prompt_hash)
    
    # Determine initial status (default to dev)
    status = "dev"
    
    # Create version
    version = ConfigVersion(
        config_id=config_id,
        parent_id=parent_id,
        created_at=datetime.utcnow().isoformat() + "Z",
        author=author,
        change_reason=change_reason,
        config=config,
        prompt=prompt_config,
        hashes=hashes,
        status=status
    )
    
    # Save to file
    version_file = _get_version_file(config_id)
    with open(version_file, 'w') as f:
        json.dump(version.dict(), f, indent=2)
    
    logger.info(f"Created config version {config_id}")
    return version


def get_pointers() -> Pointers:
    """Get current pointers."""
    _ensure_registry_dirs()
    
    if POINTERS_FILE.exists():
        try:
            with open(POINTERS_FILE, 'r') as f:
                data = json.load(f)
            return Pointers(**data)
        except Exception as e:
            logger.warning(f"Failed to load pointers: {e}")
    
    # Return defaults if file doesn't exist
    return Pointers(dev="rag_dev_v1", staging="rag_staging_v1", prod="rag_prod_v1", prod_previous=None)


def set_pointer(env: str, config_id: str) -> None:
    """
    Set pointer for environment.
    
    Args:
        env: Environment (dev, staging, or prod)
        config_id: Config ID to point to
    """
    _ensure_registry_dirs()
    
    if env not in ("dev", "staging", "prod"):
        raise ValueError(f"Invalid environment: {env}")
    
    # Validate config exists
    if get_version(config_id) is None:
        raise ValueError(f"Config {config_id} does not exist")
    
    # Load current pointers
    pointers = get_pointers()
    
    # Update pointer
    if env == "dev":
        pointers.dev = config_id
    elif env == "staging":
        pointers.staging = config_id
    elif env == "prod":
        pointers.prod = config_id
    
    # Save pointers
    with open(POINTERS_FILE, 'w') as f:
        json.dump(pointers.dict(), f, indent=2)
    
    logger.info(f"Set {env} pointer to {config_id}")


def get_pointer(env: str) -> Optional[str]:
    """
    Get pointer for environment.
    
    Args:
        env: Environment (dev, staging, or prod)
    
    Returns:
        Config ID or None
    """
    pointers = get_pointers()
    if env == "dev":
        return pointers.dev
    elif env == "staging":
        return pointers.staging
    elif env == "prod":
        return pointers.prod
    else:
        raise ValueError(f"Invalid environment: {env}")


def get_prod_history(n: int = 10) -> List[Dict]:
    """
    Get last N production history events.
    
    Args:
        n: Number of events to return
    
    Returns:
        List of history event dicts
    """
    _ensure_registry_dirs()
    
    if not HISTORY_FILE.exists():
        return []
    
    events = []
    try:
        with open(HISTORY_FILE, 'r') as f:
            lines = f.readlines()
            # Read last n lines (reverse order)
            for line in reversed(lines[-n:]):
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read history: {e}")
    
    return events

