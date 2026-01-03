"""
History logging for promotion and rollback events.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from registry.models import HistoryEvent

logger = logging.getLogger(__name__)

HISTORY_FILE = Path(__file__).parent.parent / "configs" / "registry" / "history.jsonl"


def log_event(event_type: str, from_config: str, to_config: str, actor: str, reason: Optional[str] = None) -> None:
    """
    Append event to history log.
    
    Args:
        event_type: "PROMOTE" or "ROLLBACK"
        from_config: Source config ID
        to_config: Target config ID
        actor: Actor who performed the action
        reason: Optional reason
    """
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    event = HistoryEvent(
        ts=datetime.utcnow().isoformat() + "Z",
        event=event_type,
        from_config=from_config,
        to_config=to_config,
        by=actor,
        reason=reason
    )
    
    # Append to JSONL file
    with open(HISTORY_FILE, 'a') as f:
        f.write(event.json(by_alias=True) + '\n')
    
    logger.info(f"Logged {event_type} event: {from_config} -> {to_config} by {actor}")

