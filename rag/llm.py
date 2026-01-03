"""
LLM integration with OpenAI API and stub fallback.
"""
import logging
import os
from typing import List

logger = logging.getLogger(__name__)

# Try to import openai, but don't fail if it's not available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not available. Will use stub fallback.")


def format_chunks_for_prompt(chunks: List[dict]) -> str:
    """Format retrieved chunks for inclusion in LLM prompt."""
    formatted = []
    for chunk in chunks:
        chunk_str = f"[{chunk['doc_id']}#{chunk['chunk_id']} | {chunk['timestamp']}]\n{chunk['text']}"
        formatted.append(chunk_str)
    return "\n\n---\n\n".join(formatted)


def call_openai_llm(query: str, chunks: List[dict]) -> str:
    """Call OpenAI API to generate answer."""
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Format chunks
    context = format_chunks_for_prompt(chunks)
    
    # Construct prompt
    system_prompt = "Answer using only the provided context. If context is insufficient, say you are not sure."
    user_prompt = f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"
    
    try:
        # Initialize client - use only api_key to avoid version conflicts
        # The 'proxies' error often comes from env vars or older library versions
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        answer = response.choices[0].message.content.strip()
        logger.info("Successfully generated answer using OpenAI")
        return answer
    except TypeError as e:
        # Handle version-specific errors (e.g., 'proxies' argument)
        error_msg = str(e)
        if "unexpected keyword argument" in error_msg:
            logger.error(f"OpenAI client version incompatibility: {error_msg}")
            logger.error("This usually means the openai library version is incompatible.")
            logger.error("Try: pip install --upgrade 'openai>=1.12.0'")
            raise ValueError(f"OpenAI client error: {error_msg}. Upgrade with: pip install --upgrade 'openai>=1.12.0'")
        raise
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        raise


def generate_stub_answer(query: str, chunks: List[dict]) -> str:
    """
    Generate a stub answer when OpenAI is not available.
    Uses a simple heuristic: echo relevant chunks with a basic summary.
    """
    logger.info("Using stub answer generator (OpenAI not available)")
    
    if not chunks:
        return "I don't have enough information to answer this query."
    
    # Check for conflicting information (same topic, different policies)
    # Look for chunks with similar doc_ids but different timestamps/content
    conflicting_chunks = []
    for i, chunk in enumerate(chunks):
        for j, other_chunk in enumerate(chunks[i+1:], start=i+1):
            # Check if they're about the same topic but have different policies
            if (chunk['doc_id'] != other_chunk['doc_id'] and 
                ('security' in chunk['doc_id'].lower() or 'policy' in chunk['doc_id'].lower()) and
                ('security' in other_chunk['doc_id'].lower() or 'policy' in other_chunk['doc_id'].lower())):
                # Check for conflicting language (one says "must not", other says "may")
                text1 = chunk['text'].lower()
                text2 = other_chunk['text'].lower()
                if (('must not' in text1 or 'cannot' in text1) and ('may' in text2 or 'can' in text2)) or \
                   (('must not' in text2 or 'cannot' in text2) and ('may' in text1 or 'can' in text1)):
                    conflicting_chunks = [chunk, other_chunk]
                    break
        if conflicting_chunks:
            break
    
    # Extract key information from query and chunks
    query_lower = query.lower()
    top_chunk = chunks[0]
    
    # Handle conflicts
    if conflicting_chunks:
        # Sort by timestamp (newer first)
        sorted_conflicts = sorted(conflicting_chunks, key=lambda x: x['timestamp'], reverse=True)
        newer = sorted_conflicts[0]
        older = sorted_conflicts[1]
        
        answer = f"⚠️ CONFLICTING INFORMATION DETECTED:\n\n"
        answer += f"Current Policy ({newer['timestamp'][:10]}): {newer['text']}\n\n"
        answer += f"Legacy Policy ({older['timestamp'][:10]}): {older['text']}\n\n"
        answer += f"Note: The current policy (newer) should take precedence."
    elif "yes" in query_lower or "no" in query_lower or "can" in query_lower or "may" in query_lower:
        # Try to extract yes/no signals from chunks
        text_lower = top_chunk['text'].lower()
        if "must not" in text_lower or "cannot" in text_lower or "non-refundable" in text_lower:
            answer = f"Based on the retrieved information: No. {top_chunk['text']}"
        elif "must" in text_lower or "required" in text_lower:
            answer = f"Based on the retrieved information: Yes, but with conditions. {top_chunk['text']}"
        else:
            answer = f"Based on the retrieved information: {top_chunk['text']}"
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:150]}..."
    elif "what" in query_lower or "how" in query_lower or "when" in query_lower:
        # Information-seeking question
        answer = f"Based on the retrieved documents: {top_chunk['text']}"
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:200]}..."
    else:
        # Generic answer
        answer = f"Based on the retrieved context: {top_chunk['text']}"
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:200]}..."
    
    # Add note that this is a stub
    answer += "\n\n[Note: This is a stub answer. Set OPENAI_API_KEY in .env to use real LLM for better conflict resolution.]"
    
    return answer


def generate_answer(query: str, chunks: List[dict]) -> str:
    """
    Generate an answer using OpenAI if available, otherwise use stub.
    
    Args:
        query: User query
        chunks: Retrieved chunks with metadata
    
    Returns:
        Generated answer string
    """
    # Try OpenAI first if available and key is set
    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            return call_openai_llm(query, chunks)
        except Exception as e:
            logger.warning(f"Failed to use OpenAI, falling back to stub: {e}")
            return generate_stub_answer(query, chunks)
    else:
        return generate_stub_answer(query, chunks)

