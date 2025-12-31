import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv() 

@dataclass
class Settings:
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "documents"

    HF_SPACE_URL: str = ""
    HF_SPACE_ENDPOINT: str = ""
    HF_TOKEN: Optional[str] = None

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5

    LLM_TIMEOUT: int = 120
    QDRANT_TIMEOUT: int = 10

    HNSW_M: int = 16
    HNSW_EF_CONSTRUCT: int = 100

    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    SUPPORTED_FILE_TYPES: list = None

    ENABLE_HYBRID_SEARCH: bool = True
    DENSE_WEIGHT: float = 0.7
    SPARSE_WEIGHT: float = 0.3
    HYBRID_TOP_K: int = 20
    
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    
    ENABLE_RERANKING: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_K: int = 10
    
    ENABLE_COMPRESSION: bool = True
    DEDUP_THRESHOLD: float = 0.95
    REDUNDANCY_THRESHOLD: float = 0.85
    FINAL_TOP_K: int = 5

    ENABLE_QUERY_REWRITING: bool = True
    ENABLE_HYDE: bool = False

    
    ENABLE_GRAPH_RAG: bool = True
    NEO4J_URI: str = ""
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = ""
    GRAPH_TOP_K: int = 3
    GRAPH_WEIGHT: float = 0.3 

    def __post_init__(self):
        # Load from environment
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        self.QDRANT_COLLECTION_NAME = os.getenv(
            "QDRANT_COLLECTION_NAME", self.QDRANT_COLLECTION_NAME
        )

        self.HF_SPACE_URL = os.getenv("HF_SPACE_URL")
        self.HF_SPACE_ENDPOINT = os.getenv(
            "HF_SPACE_ENDPOINT", self.HF_SPACE_ENDPOINT
        )
        self.HF_TOKEN = os.getenv("HF_TOKEN")

        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", self.CHUNK_SIZE))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", self.CHUNK_OVERLAP))
        self.TOP_K = int(os.getenv("TOP_K", self.TOP_K))

        self.ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "True").lower() == "true"
        self.ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "True").lower() == "true"
        self.ENABLE_COMPRESSION = os.getenv("ENABLE_COMPRESSION", "True").lower() == "true"
        
        self.ENABLE_QUERY_REWRITING = os.getenv("ENABLE_QUERY_REWRITING", "True").lower() == "true"
        self.ENABLE_HYDE = os.getenv("ENABLE_HYDE", "False").lower() == "true"
        
        self.ENABLE_GRAPH_RAG = os.getenv("ENABLE_GRAPH_RAG", "True").lower() == "true"
        self.NEO4J_URI = os.getenv("NEO4J_URI", "")
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

        if self.SUPPORTED_FILE_TYPES is None:
            self.SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".md"]
        
        self._validate_settings()
    
    def _validate_settings(self):
        import logging
        logger = logging.getLogger(__name__)
        
        if not self.QDRANT_URL:
            raise ValueError("❌ QDRANT_URL is required! Set it in .env file")
        
        if not self.QDRANT_API_KEY:
            raise ValueError("❌ QDRANT_API_KEY is required! Set it in .env file")
        
        if not self.HF_SPACE_URL:
            raise ValueError("❌ HF_SPACE_URL is required! Set it in .env file")
        
        if self.ENABLE_GRAPH_RAG:
            if not self.NEO4J_URI:
                logger.warning("⚠️  Graph RAG enabled but NEO4J_URI not set - disabling Graph RAG")
                self.ENABLE_GRAPH_RAG = False
            elif not self.NEO4J_PASSWORD:
                logger.warning("⚠️  NEO4J_PASSWORD not set - Graph RAG may fail")
        
        logger.info("=" * 50)
        logger.info("🔧 Configuration Loaded:")
        logger.info(f"  ✅ Qdrant: {self.QDRANT_URL}")
        logger.info(f"  ✅ LLM: {self.HF_SPACE_URL}")
        logger.info(f"  📊 Hybrid Search: {self.ENABLE_HYBRID_SEARCH}")
        logger.info(f"  🎯 Reranking: {self.ENABLE_RERANKING}")
        logger.info(f"  🧠 Query Intelligence: {self.ENABLE_QUERY_REWRITING}")
        logger.info(f"  🕸️  Graph RAG: {self.ENABLE_GRAPH_RAG}")
        if self.ENABLE_GRAPH_RAG:
            logger.info(f"  🗄️  Neo4j: {self.NEO4J_URI}")
        logger.info("=" * 50)


settings = Settings()