"""
FastAPI application for RAG MVP Day 1.
"""
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load environment variables from .env file
load_dotenv()

from rag.ingest import initialize_rag_index
from rag.llm import generate_answer
from rag.retrieve import retrieve
from rag.schemas import AnswerRequest, AnswerResponse, ChunkInfo, DocsResponse, DocumentInfo, RetrievalResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG MVP - Day 1",
    description="A minimal RAG system with naive retrieval and LLM answering",
    version="0.1.0"
)

# Global state: initialized on startup
vector_index = None
documents = []


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG index on startup."""
    global vector_index, documents
    
    try:
        logger.info("Initializing RAG index...")
        # Get path to docs.json
        script_dir = Path(__file__).parent
        docs_path = script_dir / "data" / "docs.json"
        
        vector_index, documents = initialize_rag_index(str(docs_path))
        logger.info(f"RAG system initialized with {len(documents)} documents")
    except Exception as e:
        logger.error(f"Failed to initialize RAG index: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "rag-mvp"}


@app.get("/docs", response_model=DocsResponse)
async def get_docs():
    """Get information about loaded documents."""
    if not documents:
        raise HTTPException(status_code=503, detail="Documents not loaded")
    
    doc_infos = [
        DocumentInfo(
            doc_id=doc["doc_id"],
            title=doc["title"],
            timestamp=doc["timestamp"]
        )
        for doc in documents
    ]
    
    return DocsResponse(count=len(documents), documents=doc_infos)


@app.post("/answer", response_model=AnswerResponse)
async def answer_query(request: AnswerRequest):
    """
    Answer a query using RAG:
    1. Retrieve top-k relevant chunks
    2. Generate answer using LLM with retrieved context
    """
    if vector_index is None:
        raise HTTPException(status_code=503, detail="RAG index not initialized")
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = retrieve(vector_index, request.query, request.top_k)
        
        # Generate answer
        answer = generate_answer(request.query, retrieved_chunks)
        
        # Format response
        chunk_infos = [
            ChunkInfo(
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                timestamp=chunk["timestamp"],
                similarity=chunk["similarity"],
                text=chunk["text"]
            )
            for chunk in retrieved_chunks
        ]
        
        response = AnswerResponse(
            query=request.query,
            answer=answer,
            retrieval=RetrievalResult(
                top_k=request.top_k,
                chunks=chunk_infos
            )
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

