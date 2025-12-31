import time
import logging
import uuid
from typing import List, Dict, Any
from pathlib import Path

from langchain_core.documents import Document
from fastapi import UploadFile

from config import settings
from rag_core import (
    EmbeddingEncoder,
    QdrantVectorDB,
    HFSpaceLLMClient,
    DocumentChunker,
    DocumentLoader,
    RAGPrompts,
    BM25Retriever,
    CrossEncoderReranker,
    ContextCompressor,
    HybridRetriever,
    QueryRewriter,
    HyDEGenerator,
    Neo4jGraphManager,
    GraphDocumentBuilder,
    GraphCypherRetriever
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG orchestration pipeline - Phase 4 (Neo4j)"""
    
    def __init__(self):
        logger.info("Initializing RAG Pipeline (Phase 4 - Neo4j Graph)...")
        
        # Phase 1 components
        self.encoder = EmbeddingEncoder(settings.EMBEDDING_MODEL)
        self.vectordb = QdrantVectorDB(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            collection_name=settings.QDRANT_COLLECTION_NAME
        )
        self.llm = HFSpaceLLMClient(
            base_url=settings.HF_SPACE_URL,
            endpoint=settings.HF_SPACE_ENDPOINT,
            token=settings.HF_TOKEN
        )
        self.chunker = DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
        self.prompt_template = RAGPrompts.get_qa_prompt()
        
        # Phase 2 components
        self.bm25_retriever = BM25Retriever(
            k1=settings.BM25_K1,
            b=settings.BM25_B
        )
        
        self.reranker = None
        if settings.ENABLE_RERANKING:
            self.reranker = CrossEncoderReranker(settings.RERANKER_MODEL)
        
        self.compressor = None
        if settings.ENABLE_COMPRESSION:
            self.compressor = ContextCompressor(
                encoder=self.encoder,
                dedup_threshold=settings.DEDUP_THRESHOLD,
                redundancy_threshold=settings.REDUNDANCY_THRESHOLD
            )
        
        if settings.ENABLE_HYBRID_SEARCH:
            self.hybrid_retriever = HybridRetriever(
                dense_retriever=self.vectordb,
                sparse_retriever=self.bm25_retriever,
                encoder=self.encoder,
                reranker=self.reranker,
                compressor=self.compressor,
                dense_weight=settings.DENSE_WEIGHT,
                sparse_weight=settings.SPARSE_WEIGHT
            )
        
        # Phase 3 components
        self.query_rewriter = None
        if settings.ENABLE_QUERY_REWRITING:
            self.query_rewriter = QueryRewriter(self.llm)
        
        self.hyde_generator = None
        if settings.ENABLE_HYDE:
            self.hyde_generator = HyDEGenerator(self.llm, self.encoder)
        
        # ============ Phase 4 components - Neo4j ============
        
        self.neo4j_manager = None
        self.graph_builder = None
        self.graph_retriever = None
        
        if settings.ENABLE_GRAPH_RAG and settings.NEO4J_URI:
            # Connect to Neo4j
            self.neo4j_manager = Neo4jGraphManager(
                uri=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD
            )
            
            # Initialize graph document builder
            self.graph_builder = GraphDocumentBuilder(self.llm)
            
            # Initialize graph retriever
            self.graph_retriever = GraphCypherRetriever(
                neo4j_manager=self.neo4j_manager,
                llm_client=self.llm,
                vectordb=self.vectordb,
                encoder=self.encoder,
                graph_weight=settings.GRAPH_WEIGHT
            )
        
        logger.info("RAG Pipeline (Phase 4 - Neo4j) initialized successfully")
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Ingest documents into vector DB and Neo4j graph
        """
        all_langchain_docs = []
        
        for doc in documents:
            chunked_docs = self.chunker.chunk_document(
                text=doc["text"],
                document_id=doc["document_id"],
                source=doc["source"],
                metadata=doc.get("metadata", {})
            )
            all_langchain_docs.extend(chunked_docs)
        
        texts = [doc.page_content for doc in all_langchain_docs]
        embeddings = self.encoder.encode_batch(texts)
        
        # Upload to Qdrant
        self.vectordb.upsert_documents(all_langchain_docs, embeddings)
        
        # Index for BM25
        if settings.ENABLE_HYBRID_SEARCH:
            self.bm25_retriever.index_documents(all_langchain_docs)
        
        graph_stats = None
        if settings.ENABLE_GRAPH_RAG and self.graph_builder and self.neo4j_manager:
            logger.info("Building Neo4j knowledge graph...")
            try:
                graph_documents = self.graph_builder.build_graph_documents(all_langchain_docs)
                
                if graph_documents:
                    self.neo4j_manager.add_graph_documents(graph_documents)
                    
                    # Get graph statistics
                    schema = self.neo4j_manager.get_schema()
                    graph_stats = {
                        "graph_documents_created": len(graph_documents),
                        "schema": schema
                    }
                    logger.info(f"Neo4j graph updated with {len(graph_documents)} graph documents")
                
            except Exception as e:
                logger.error(f"Failed to build Neo4j graph: {e}")
                graph_stats = {"error": str(e)}
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "total_chunks": len(all_langchain_docs),
            "collection_name": settings.QDRANT_COLLECTION_NAME,
            "phase": "Phase 4 - Neo4j Graph-Augmented RAG",
            "graph_stats": graph_stats
        }
    
    async def ingest_file(self, file: UploadFile, document_id: str = None) -> Dict[str, Any]:
        """Ingest file - includes Neo4j graph building"""
        text, file_extension = await DocumentLoader.load_file(file)
        
        if not document_id:
            document_id = f"{Path(file.filename).stem}_{uuid.uuid4().hex[:8]}"
        
        result = self.ingest_documents([{
            "document_id": document_id,
            "source": file.filename,
            "text": text,
            "metadata": {
                "file_type": file_extension,
                "file_size": len(text),
                "upload_timestamp": time.time()
            }
        }])
        
        result["filename"] = file.filename
        result["file_type"] = file_extension
        return result
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Execute RAG query pipeline
        Phase 4: Neo4j graph-augmented retrieval
        """
        start_time = time.time()
        original_query = query
        
        # Phase 3: Query Intelligence
        if settings.ENABLE_QUERY_REWRITING and self.query_rewriter:
            query = self.query_rewriter.rewrite(query)
        
        # Phase 2/3: Get initial vector results
        if settings.ENABLE_HYDE and self.hyde_generator:
            hyde_embedding = self.hyde_generator.get_hyde_embedding(query)
            initial_contexts = self.vectordb.search(hyde_embedding, settings.HYBRID_TOP_K)
            retrieval_method = "hyde_phase3"
        elif settings.ENABLE_HYBRID_SEARCH:
            initial_contexts = self.hybrid_retriever.search(
                query=query,
                top_k=settings.HYBRID_TOP_K,
                enable_reranking=settings.ENABLE_RERANKING,
                enable_compression=False
            )
            retrieval_method = "hybrid_phase2"
        else:
            query_embedding = self.encoder.encode(query)
            initial_contexts = self.vectordb.search(query_embedding, settings.TOP_K)
            retrieval_method = "dense_phase1"
        
        # ============ Phase 4: Neo4j Graph Enhancement ============
        if settings.ENABLE_GRAPH_RAG and self.graph_retriever:
            contexts = self.graph_retriever.search_with_graph(
                query=query,
                vector_results=initial_contexts,
                top_k=settings.FINAL_TOP_K
            )
            retrieval_method = f"{retrieval_method}_neo4j_graph"
        else:
            contexts = initial_contexts[:settings.FINAL_TOP_K]
        
        # Apply compression if enabled
        if settings.ENABLE_COMPRESSION and self.compressor and len(contexts) > settings.FINAL_TOP_K:
            contexts = self.compressor.compress(contexts, settings.FINAL_TOP_K)
        
        # Rest is same
        if not contexts:
            return {
                "query": original_query,
                "processed_query": query if query != original_query else None,
                "answer": "No relevant information found in the knowledge base.",
                "contexts": [],
                "retrieval_count": 0,
                "retrieval_method": retrieval_method,
                "response_time_ms": int((time.time() - start_time) * 1000)
            }
        
        # Format contexts for prompt
        formatted_contexts = RAGPrompts.format_contexts(contexts)
        
        # Generate prompt
        prompt = self.prompt_template.format(
            context=formatted_contexts,
            question=query
        )
        
        # Generate answer
        answer = self.llm.generate(prompt)
        
        response_time = int((time.time() - start_time) * 1000)
        
        return {
            "query": original_query,
            "processed_query": query if query != original_query else None,
            "answer": answer,
            "contexts": contexts,
            "retrieval_count": len(contexts),
            "retrieval_method": retrieval_method,
            "response_time_ms": response_time
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all services"""
        health = {
            "status": "healthy",
            "services": {
                "embeddings": "unknown",
                "qdrant": "unknown",
                "llm": "unknown",
                "bm25": "unknown",
                "reranker": "unknown",
                "query_intelligence": "unknown",
                "neo4j_graph": "unknown"
            }
        }
        
        # Check embeddings
        try:
            test_embedding = self.encoder.encode("test")
            if len(test_embedding) == settings.EMBEDDING_DIM:
                health["services"]["embeddings"] = "loaded"
        except Exception as e:
            health["services"]["embeddings"] = f"error: {str(e)}"
            health["status"] = "unhealthy"
        
        # Check Qdrant
        try:
            self.vectordb.client.get_collection(settings.QDRANT_COLLECTION_NAME)
            health["services"]["qdrant"] = "connected"
        except Exception as e:
            health["services"]["qdrant"] = f"error: {str(e)}"
            health["status"] = "unhealthy"
        
        # Check LLM
        try:
            health["services"]["llm"] = "reachable"
        except Exception as e:
            health["services"]["llm"] = f"error: {str(e)}"
        
        # Check BM25
        try:
            if self.bm25_retriever.bm25 is not None:
                health["services"]["bm25"] = "indexed"
            else:
                health["services"]["bm25"] = "not_indexed"
        except Exception as e:
            health["services"]["bm25"] = f"error: {str(e)}"
        
        # Check Reranker
        try:
            if self.reranker:
                health["services"]["reranker"] = "loaded"
            else:
                health["services"]["reranker"] = "disabled"
        except Exception as e:
            health["services"]["reranker"] = f"error: {str(e)}"
        
        # Check Query Intelligence
        try:
            active_features = []
            if self.query_rewriter:
                active_features.append("query-rewriter")
            if self.hyde_generator:
                active_features.append("hyde")
            
            if active_features:
                health["services"]["query_intelligence"] = f"active: {', '.join(active_features)}"
            else:
                health["services"]["query_intelligence"] = "disabled"
        except Exception as e:
            health["services"]["query_intelligence"] = f"error: {str(e)}"
        
        # Check Neo4j
        try:
            if self.neo4j_manager and self.neo4j_manager.graph:
                schema_preview = str(self.neo4j_manager.get_schema())[:100]
                health["services"]["neo4j_graph"] = f"connected: {schema_preview}..."
            else:
                health["services"]["neo4j_graph"] = "disabled"
        except Exception as e:
            health["services"]["neo4j_graph"] = f"error: {str(e)}"
        
        return health