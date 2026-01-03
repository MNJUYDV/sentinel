"""
Document ingestion, chunking, and index building.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class DocumentChunk:
    """Represents a chunk of a document."""
    def __init__(self, doc_id: str, chunk_id: str, text: str, timestamp: str):
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.text = text
        self.timestamp = timestamp


def load_documents(docs_path: str) -> List[Dict]:
    """Load documents from JSON file."""
    try:
        with open(docs_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        logger.info(f"Loaded {len(docs)} documents from {docs_path}")
        return docs
    except FileNotFoundError:
        logger.error(f"Documents file not found: {docs_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in documents file: {e}")
        raise


def chunk_document(doc: Dict) -> List[DocumentChunk]:
    """
    Chunk a document by splitting on blank lines (paragraphs).
    Each chunk gets a chunk_id = "<doc_id>#p<idx>"
    """
    doc_id = doc["doc_id"]
    text = doc["text"]
    timestamp = doc["timestamp"]
    
    # Split by blank lines (double newline or more)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for idx, para in enumerate(paragraphs):
        chunk_id = f"{doc_id}#p{idx}"
        chunks.append(DocumentChunk(doc_id, chunk_id, para, timestamp))
    
    return chunks


def chunk_all_documents(docs: List[Dict]) -> List[DocumentChunk]:
    """Chunk all documents."""
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks


class VectorIndex:
    """Manages the vector index and embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding model."""
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[DocumentChunk] = []
        logger.info(f"Model dimension: {self.dimension}")
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index from chunks."""
        self.chunks = chunks
        
        # Generate embeddings
        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index (L2 distance, but we'll use cosine similarity)
        # Since embeddings are normalized, L2 distance is equivalent to cosine distance
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for similar chunks.
        Returns list of (chunk, similarity) tuples sorted by similarity (descending).
        """
        if self.index is None or len(self.chunks) == 0:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        k = min(top_k, len(self.chunks))
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert distances to similarities (for normalized embeddings, similarity = 1 - distance^2 / 2)
        # Or we can use: similarity = 1 / (1 + distance) for a simple transformation
        # Actually, for normalized L2: cosine_similarity = 1 - (distance^2 / 2)
        # But let's use a simpler transformation that's more intuitive
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                # For normalized embeddings, L2 distance relates to cosine similarity
                # cosine_sim = 1 - (distance^2 / 2) when embeddings are normalized
                similarity = max(0.0, 1.0 - (dist ** 2) / 2.0)
                results.append((self.chunks[idx], similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def initialize_rag_index(docs_path: str = None) -> Tuple[VectorIndex, List[Dict]]:
    """
    Initialize the RAG system by loading documents, chunking, and building index.
    
    Args:
        docs_path: Path to documents JSON file. If None, uses default path.
    
    Returns:
        Tuple of (VectorIndex, list of original documents)
    """
    if docs_path is None:
        # Default to data/docs.json relative to project root
        script_dir = Path(__file__).parent.parent
        docs_path = script_dir / "data" / "docs.json"
        docs_path = str(docs_path)
    
    # Load documents
    docs = load_documents(docs_path)
    
    # Chunk documents
    chunks = chunk_all_documents(docs)
    
    # Build index
    index = VectorIndex()
    index.build_index(chunks)
    
    return index, docs

