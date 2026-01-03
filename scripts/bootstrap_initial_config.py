#!/usr/bin/env python3
"""
Bootstrap script to create initial config versions for dev, staging, and prod.
Run this once to set up the registry with initial configs.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from registry.store import create_version, set_pointer
from rag.config import Config

# Default prompt (matches current LLM implementation)
DEFAULT_PROMPT = {
    "system_prompt": """You are a helpful assistant that answers questions using only the provided context.
You MUST respond with ONLY a valid JSON object, no additional text.

Your response must be exactly this format:
{
  "answer": "your answer here",
  "citations": ["chunk_id1", "chunk_id2"]
}

Rules:
1. The "citations" array must contain ONLY chunk IDs from the allowed list provided.
2. If the context is insufficient to answer, set answer to indicate this (e.g., "Insufficient context to answer this question") and set citations to [].
3. Only cite chunks that directly support claims in your answer.
4. If you cannot support the answer with citations, indicate insufficiency rather than making unsupported claims.""",
    "user_template": "Query: {query}\n\nContext:\n{context}\n\nAllowed citation IDs: {allowed_citations}\n\nProvide your answer as JSON only:"
}

# Default config (matches baseline.json)
DEFAULT_CONFIG = {
    "top_k": 5,
    "freshness_days": 90,
    "freshness_days_high_risk": 30,
    "confidence_threshold": 0.60,
    "confidence_threshold_high_risk": 0.70,
    "min_chunks": 2
}


def main():
    print("Bootstrapping initial config versions...")
    
    # Create initial configs for each environment
    dev_version = create_version(
        parent_id=None,
        author="system",
        change_reason="Initial dev config",
        config=DEFAULT_CONFIG,
        prompt=DEFAULT_PROMPT,
        config_id="rag_dev_v1"
    )
    print(f"Created dev config: {dev_version.config_id}")
    
    staging_version = create_version(
        parent_id=None,
        author="system",
        change_reason="Initial staging config",
        config=DEFAULT_CONFIG,
        prompt=DEFAULT_PROMPT,
        config_id="rag_staging_v1"
    )
    print(f"Created staging config: {staging_version.config_id}")
    
    prod_version = create_version(
        parent_id=None,
        author="system",
        change_reason="Initial prod config",
        config=DEFAULT_CONFIG,
        prompt=DEFAULT_PROMPT,
        config_id="rag_prod_v1"
    )
    print(f"Created prod config: {prod_version.config_id}")
    
    # Set pointers
    set_pointer("dev", dev_version.config_id)
    set_pointer("staging", staging_version.config_id)
    set_pointer("prod", prod_version.config_id)
    
    print("\nInitial config versions created and pointers set!")
    print("You can now use the config registry API endpoints.")


if __name__ == "__main__":
    main()

