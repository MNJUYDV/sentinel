"""
LLM integration with OpenAI API and stub fallback.
Returns structured JSON with answer and citations.
"""
import json
import logging
import os
import re
from typing import Dict, List, Optional

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
        chunk_str = f"[{chunk['chunk_id']} | {chunk['timestamp']}]\n{chunk['text']}"
        formatted.append(chunk_str)
    return "\n\n---\n\n".join(formatted)


def get_allowed_citation_ids(chunks: List[dict]) -> List[str]:
    """Extract allowed citation IDs from retrieved chunks."""
    return [chunk["chunk_id"] for chunk in chunks]


def call_openai_llm(query: str, chunks: List[dict]) -> Dict[str, any]:
    """
    Call OpenAI API to generate structured answer with citations.
    
    Returns:
        Dict with keys: "answer" (str) and "citations" (List[str])
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI library not installed")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Format chunks
    context = format_chunks_for_prompt(chunks)
    
    # Get allowed citation IDs
    allowed_citations = get_allowed_citation_ids(chunks)
    allowed_citations_str = ", ".join(allowed_citations)
    
    # Construct prompts
    system_prompt = """You are a helpful assistant that answers questions using only the provided context.
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
4. If you cannot support the answer with citations, indicate insufficiency rather than making unsupported claims."""

    user_prompt = f"""Query: {query}

Context:
{context}

Allowed citation IDs: {allowed_citations_str}

Provide your answer as JSON only:"""

    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=600,
            response_format={"type": "json_object"}
        )
        
        # Parse JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON if there's extra text
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks or other formatting
            json_match = re.search(r'\{[^{}]*"answer"[^{}]*"citations"[^{}]*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError(f"Could not parse JSON from response: {content[:200]}")
        
        # Validate structure
        if "answer" not in result or "citations" not in result:
            raise ValueError(f"Response missing required fields. Got: {list(result.keys())}")
        
        if not isinstance(result["answer"], str):
            raise ValueError(f"Answer must be a string, got: {type(result['answer'])}")
        
        if not isinstance(result["citations"], list):
            raise ValueError(f"Citations must be a list, got: {type(result['citations'])}")
        
        # Ensure citations are strings
        result["citations"] = [str(c) for c in result["citations"]]
        
        logger.info(f"Successfully generated structured answer with {len(result['citations'])} citations")
        return result
        
    except TypeError as e:
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


def generate_stub_answer(query: str, chunks: List[dict]) -> Dict[str, any]:
    """
    Generate a stub answer with citations when OpenAI is not available.
    Returns structured JSON with valid citations that reference retrieved chunks.
    """
    logger.info("Using stub answer generator (OpenAI not available)")
    
    if not chunks:
        return {
            "answer": "I don't have enough information to answer this query based on the retrieved documents.",
            "citations": []
        }
    
    # Get allowed citation IDs
    allowed_citations = get_allowed_citation_ids(chunks)
    top_chunk = chunks[0]
    
    # Extract key information from query and chunks
    query_lower = query.lower()
    
    # Generate answer and always include at least one citation
    citations = [top_chunk["chunk_id"]]
    
    # Check for conflicting information
    conflicting_chunks = []
    for i, chunk in enumerate(chunks):
        for j, other_chunk in enumerate(chunks[i+1:], start=i+1):
            if (chunk['doc_id'] != other_chunk['doc_id'] and 
                ('security' in chunk['doc_id'].lower() or 'policy' in chunk['doc_id'].lower()) and
                ('security' in other_chunk['doc_id'].lower() or 'policy' in other_chunk['doc_id'].lower())):
                text1 = chunk['text'].lower()
                text2 = other_chunk['text'].lower()
                if (('must not' in text1 or 'cannot' in text1) and ('may' in text2 or 'can' in text2)) or \
                   (('must not' in text2 or 'cannot' in text2) and ('may' in text1 or 'can' in text1)):
                    conflicting_chunks = [chunk, other_chunk]
                    break
        if conflicting_chunks:
            break
    
    # Generate answer based on context
    if conflicting_chunks:
        sorted_conflicts = sorted(conflicting_chunks, key=lambda x: x['timestamp'], reverse=True)
        newer = sorted_conflicts[0]
        older = sorted_conflicts[1]
        
        answer = f"⚠️ CONFLICTING INFORMATION DETECTED:\n\n"
        answer += f"Current Policy ({newer['timestamp'][:10]}): {newer['text']}\n\n"
        answer += f"Legacy Policy ({older['timestamp'][:10]}): {older['text']}\n\n"
        answer += f"Note: The current policy (newer) should take precedence."
        citations = [newer["chunk_id"], older["chunk_id"]]
    elif "yes" in query_lower or "no" in query_lower or "can" in query_lower or "may" in query_lower:
        text_lower = top_chunk['text'].lower()
        if "must not" in text_lower or "cannot" in text_lower or "non-refundable" in text_lower:
            answer = f"Based on the retrieved information: No. {top_chunk['text']}"
        elif "must" in text_lower or "required" in text_lower:
            answer = f"Based on the retrieved information: Yes, but with conditions. {top_chunk['text']}"
        else:
            answer = f"Based on the retrieved information: {top_chunk['text']}"
        
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:150]}..."
            citations.append(chunks[1]["chunk_id"])
    elif "what" in query_lower or "how" in query_lower or "when" in query_lower:
        answer = f"Based on the retrieved documents: {top_chunk['text']}"
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:200]}..."
            citations.append(chunks[1]["chunk_id"])
    else:
        answer = f"Based on the retrieved context: {top_chunk['text']}"
        if len(chunks) > 1:
            answer += f"\n\nAdditional context: {chunks[1]['text'][:200]}..."
            citations.append(chunks[1]["chunk_id"])
    
    # Limit citations to max 5
    citations = citations[:5]
    
    # Ensure all citations are valid
    citations = [c for c in citations if c in allowed_citations]
    
    # Add note that this is a stub
    answer += "\n\n[Note: This is a stub answer. Set OPENAI_API_KEY in .env to use real LLM.]"
    
    return {
        "answer": answer,
        "citations": citations
    }


def generate_answer(query: str, chunks: List[dict]) -> Dict[str, any]:
    """
    Generate a structured answer with citations using OpenAI if available, otherwise use stub.
    
    Args:
        query: User query
        chunks: Retrieved chunks with metadata
    
    Returns:
        Dict with keys: "answer" (str) and "citations" (List[str])
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
