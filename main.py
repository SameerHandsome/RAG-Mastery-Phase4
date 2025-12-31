import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from config import settings
from rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Phase 4 RAG System - Neo4j",
    description="Advanced RAG with Neo4j knowledge graph, query intelligence, hybrid retrieval, and reranking",
    version="4.0.0"
)

rag = RAGPipeline()

class Document(BaseModel):
    document_id: str
    source: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    documents: List[Document]

class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    total_chunks: int
    collection_name: str

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_type: str
    document_id: str
    total_chunks: int
    collection_name: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)

class QueryResponse(BaseModel):
    query: str
    answer: str
    contexts: List[Dict[str, Any]]
    retrieval_count: int
    response_time_ms: int

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    try:
        documents = [doc.dict() for doc in request.documents]
        result = rag.ingest_documents(documents)
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None)
):
    try:
        result = await rag.ingest_file(file, document_id)
        return {
            "status": result["status"],
            "filename": result["filename"],
            "file_type": result["file_type"],
            "document_id": result.get("document_id", document_id or "auto_generated"),
            "total_chunks": result["total_chunks"],
            "collection_name": result["collection_name"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = rag.query(request.query)
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        return rag.health_check()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Phase 4 RAG System - Neo4j Graph",
        "version": "4.0.0",
        "phase": "Phase 4 - Neo4j Graph-Augmented RAG",
        "features": {
            "neo4j_graph": settings.ENABLE_GRAPH_RAG,
            "query_rewriting": settings.ENABLE_QUERY_REWRITING,
            "hyde": settings.ENABLE_HYDE,
            "hybrid_search": settings.ENABLE_HYBRID_SEARCH,
            "reranking": settings.ENABLE_RERANKING,
            "compression": settings.ENABLE_COMPRESSION
        },
        "endpoints": {
            "ingest": "/ingest (POST)",
            "upload": "/upload (POST)",
            "query": "/query (POST)",
            "health": "/health (GET)",
            "docs": "/docs (GET)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Phase 4 RAG System (Neo4j)...")
    logger.info(f"Neo4j URI: {settings.NEO4J_URI}")
    logger.info(f"Graph RAG: {settings.ENABLE_GRAPH_RAG}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")