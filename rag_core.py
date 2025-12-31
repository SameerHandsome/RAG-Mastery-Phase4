import uuid
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Optional, Mapping
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import PyPDF2
import docx2txt
from fastapi import HTTPException, UploadFile
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def encode(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


class QdrantVectorDB:
    def __init__(self, url: str, api_key: str, collection_name: str):
        logger.info(f"Connecting to Qdrant Cloud: {url}")
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=settings.QDRANT_TIMEOUT
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.http.models import VectorParams, Distance
        
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            logger.info(f"Using existing collection '{self.collection_name}'")
            return

        logger.info(f"Collection '{self.collection_name}' not found. Creating...")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=Distance.COSINE
            ),
            hnsw_config={
                "m": settings.HNSW_M,
                "ef_construct": settings.HNSW_EF_CONSTRUCT
            }
        )

    def upsert_documents(self, documents: List[Document], embeddings: List[List[float]]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
            )
            for doc, embedding in zip(documents, embeddings)
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Upserted {len(points)} documents to Qdrant")

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        # This version tries multiple methods to find the right API for your installation
        try:
            # 1. Try modern query_points (preferred)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k
            )
            results = response.points
        except AttributeError:
            try:
                # 2. Try standard search
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=top_k
                )
            except AttributeError:
                # 3. Last resort: Direct HTTP API access
                from qdrant_client import models
                response = self.client.http.points_api.search_points(
                    collection_name=self.collection_name,
                    search_points=models.SearchRequest(
                        vector=query_embedding,
                        limit=top_k,
                        with_payload=True
                    )
                )
                results = response.result

        contexts = []
        for result in results:
            # Handle potential differences in result object structure
            payload = getattr(result, 'payload', {}) or {}
            metadata = payload.get("metadata", {})
            
            contexts.append({
                "text": payload.get("text", ""),
                "metadata": {
                    "document_id": metadata.get("document_id", ""),
                    "source": metadata.get("source", ""),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "score": getattr(result, 'score', 0.0)
                }
            })

        logger.info(f"Retrieved {len(contexts)} contexts from Qdrant")
        return contexts


class HFSpaceLLMClient:
    def __init__(self, base_url: str, endpoint: str, token: Optional[str] = None):
        self.url = f"{base_url}{endpoint}"
        self.token = token
        self.session = requests.Session()
        logger.info(f"Initialized LLM client for: {self.url}")
    
    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        
        payload = {
            "query": prompt  
        }
        
        try:
            logger.info("Calling LLM API...")
            response = self.session.post(
                self.url,
                json=payload,
                headers=headers,
                timeout=settings.LLM_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "")
            
            if not answer:
                answer = result.get("generated_text", result.get("text", str(result)))
            
            logger.info("LLM response received")
            return answer.strip()
            
        except requests.exceptions.Timeout:
            logger.error("LLM API timeout")
            raise HTTPException(status_code=504, detail="LLM API timeout")
        except requests.exceptions.HTTPError as e:
            logger.error(f"LLM API HTTP error: {e}")
            raise HTTPException(status_code=500, detail=f"LLM API error: {str(e)}")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    def __del__(self):
        self.session.close()


class RAGPrompts:
    @staticmethod
    def get_qa_prompt() -> PromptTemplate:
        template = """You are a helpful assistant that answers questions based strictly on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer ONLY using information from the context above
- If the context doesn't contain enough information to answer, respond with: "I cannot answer this question based on the provided context."
- Be concise and direct
- Do not add information not present in the context

Answer:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
    
    @staticmethod
    def format_contexts(contexts: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, ctx in enumerate(contexts, 1):
            formatted.append(f"[Document {i}]\n{ctx['text']}")
        return "\n\n".join(formatted)


class DocumentChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        logger.info(f"Initialized DocumentChunker with chunk_size={chunk_size}, overlap={overlap}")
    
    def chunk_document(
        self, 
        text: str, 
        document_id: str, 
        source: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        chunks = self.splitter.split_text(text)
        
        documents = []
        for chunk_index, chunk_text in enumerate(chunks):
            chunk_metadata = {
                "document_id": document_id,
                "source": source,
                "chunk_index": chunk_index,
                "total_chunks": len(chunks)
            }
            
            if metadata:
                chunk_metadata.update(metadata)
            
            doc = Document(
                page_content=chunk_text,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks from document {document_id}")
        return documents


class DocumentLoader:
    @classmethod
    async def load_file(cls, file: UploadFile) -> tuple[str, str]:
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in settings.SUPPORTED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported: {settings.SUPPORTED_FILE_TYPES}"
            )
        
        content = await file.read()
        
        if len(content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )
        
        try:
            logger.info(f"Loading {file_extension} file: {file.filename}")
            
            if file_extension == ".pdf":
                text = cls._extract_pdf(content)
            elif file_extension == ".docx":
                text = cls._extract_docx(content)
            elif file_extension == ".txt":
                text = cls._extract_txt(content)
            elif file_extension == ".md":
                text = cls._extract_txt(content)  
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"No loader available for {file_extension}"
                )
            
            logger.info(f"Successfully extracted {len(text)} characters from {file.filename}")
            return text, file_extension
            
        except Exception as e:
            logger.error(f"Error loading file {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process file: {str(e)}"
            )
    
    @staticmethod
    def _extract_pdf(content: bytes) -> str:
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = []
        for page in pdf_reader.pages:
            text.append(page.extract_text())
        return "\n\n".join(text)
    
    @staticmethod
    def _extract_docx(content: bytes) -> str:
        import io
        with io.BytesIO(content) as docx_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
                text = docx2txt.process(tmp_path)
                return text
            finally:
                Path(tmp_path).unlink()
    
    @staticmethod
    def _extract_txt(content: bytes) -> str:
        return content.decode('utf-8', errors='ignore')


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []          
        self.metadata = []        
        self.bm25 = None
        logger.info(f"Initialized BM25Retriever with k1={k1}, b={b}")
    
    def index_documents(self, documents: List[Document]):
        tokenized_corpus = []
        
        for doc in documents:
            tokens = doc.page_content.lower().split()
            tokenized_corpus.append(tokens)
            
            self.corpus.append(doc.page_content)
            self.metadata.append(doc.metadata)
        
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        logger.info(f"Indexed {len(documents)} documents for BM25")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return []
        
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  
                results.append({
                    "text": self.corpus[idx],
                    "metadata": {
                        **self.metadata[idx],
                        "score": float(scores[idx]),
                        "retrieval_type": "sparse_bm25"
                    }
                })
        
        logger.info(f"BM25 retrieved {len(results)} results")
        return results


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded successfully")
    
    def rerank(
        self, 
        query: str, 
        contexts: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        
        pairs = [[query, ctx["text"]] for ctx in contexts]
        scores = self.model.predict(pairs)
        
        for ctx, score in zip(contexts, scores):
            ctx["metadata"]["rerank_score"] = float(score)
            ctx["metadata"]["original_score"] = ctx["metadata"].get("score", 0.0)
        
        reranked = sorted(
            contexts, 
            key=lambda x: x["metadata"]["rerank_score"], 
            reverse=True
        )[:top_k]
        
        logger.info(f"Reranked {len(contexts)} contexts to top {top_k}")
        return reranked


class ContextCompressor:
    def __init__(
        self,
        encoder: 'EmbeddingEncoder',
        dedup_threshold: float = 0.95,
        redundancy_threshold: float = 0.85
    ):
        self.encoder = encoder
        self.dedup_threshold = dedup_threshold
        self.redundancy_threshold = redundancy_threshold
        logger.info(f"Initialized ContextCompressor (dedup={dedup_threshold}, redundancy={redundancy_threshold})")
    
    def compress(
        self, 
        contexts: List[Dict[str, Any]], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not contexts:
            return []
        
        deduplicated = self._deduplicate(contexts)
        logger.info(f"After deduplication: {len(deduplicated)} contexts")
        
        compressed = self._remove_redundancy(deduplicated)
        logger.info(f"After redundancy removal: {len(compressed)} contexts")
        
        final = compressed[:top_k]
        logger.info(f"Final compressed contexts: {len(final)}")
        
        return final
    
    def _deduplicate(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_texts = {}
        
        for ctx in contexts:
            text = ctx["text"].strip()
            score = ctx["metadata"].get("rerank_score", ctx["metadata"].get("score", 0))
            
            if text not in seen_texts:
                seen_texts[text] = ctx
            else:
                existing_score = seen_texts[text]["metadata"].get(
                    "rerank_score", 
                    seen_texts[text]["metadata"].get("score", 0)
                )
                if score > existing_score:
                    seen_texts[text] = ctx
        
        return list(seen_texts.values())
    
    def _remove_redundancy(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(contexts) <= 1:
            return contexts
        
        texts = [ctx["text"] for ctx in contexts]
        embeddings = np.array(self.encoder.encode_batch(texts))
        similarity_matrix = cosine_similarity(embeddings)
        
        keep_indices = []
        for i in range(len(contexts)):
            is_redundant = False
            
            for kept_idx in keep_indices:
                similarity = similarity_matrix[i][kept_idx]
                if similarity > self.redundancy_threshold:
                    ctx_score = contexts[i]["metadata"].get(
                        "rerank_score",
                        contexts[i]["metadata"].get("score", 0)
                    )
                    kept_score = contexts[kept_idx]["metadata"].get(
                        "rerank_score",
                        contexts[kept_idx]["metadata"].get("score", 0)
                    )
                    
                    if ctx_score > kept_score:
                        keep_indices.remove(kept_idx)
                        keep_indices.append(i)
                    
                    is_redundant = True
                    break
            
            if not is_redundant:
                keep_indices.append(i)
        
        return [contexts[i] for i in keep_indices]


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: QdrantVectorDB,
        sparse_retriever: BM25Retriever,
        encoder: EmbeddingEncoder,
        reranker: Optional[CrossEncoderReranker] = None,
        compressor: Optional[ContextCompressor] = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.encoder = encoder
        self.reranker = reranker
        self.compressor = compressor
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        logger.info(f"Initialized HybridRetriever (dense={dense_weight}, sparse={sparse_weight})")
    
    def search(
        self, 
        query: str, 
        top_k: int = 5,
        enable_reranking: bool = True,
        enable_compression: bool = True
    ) -> List[Dict[str, Any]]:
        query_embedding = self.encoder.encode(query)
        dense_results = self.dense_retriever.search(
            query_embedding, 
            top_k=settings.HYBRID_TOP_K
        )
        logger.info(f"Dense retrieval: {len(dense_results)} results")
        
        sparse_results = self.sparse_retriever.search(
            query, 
            top_k=settings.HYBRID_TOP_K
        )
        logger.info(f"Sparse retrieval: {len(sparse_results)} results")
        
        fused_results = self._fuse_results(dense_results, sparse_results)
        logger.info(f"After fusion: {len(fused_results)} unique results")
        
        if enable_reranking and self.reranker:
            fused_results = self.reranker.rerank(
                query, 
                fused_results, 
                top_k=settings.RERANK_TOP_K
            )
            logger.info(f"After reranking: {len(fused_results)} results")
        
        if enable_compression and self.compressor:
            fused_results = self.compressor.compress(
                fused_results, 
                top_k=top_k
            )
            logger.info(f"After compression: {len(fused_results)} results")
        else:
            fused_results = fused_results[:top_k]
        
        return fused_results
    
    def _fuse_results(
        self, 
        dense_results: List[Dict], 
        sparse_results: List[Dict]
    ) -> List[Dict[str, Any]]:
        all_results = {}
        
        for result in dense_results:
            key = f"{result['metadata']['document_id']}_{result['metadata']['chunk_index']}"
            dense_score = result["metadata"].get("score", 0)
            
            all_results[key] = {
                **result,
                "fusion_score": self.dense_weight * dense_score,
                "dense_score": dense_score,
                "sparse_score": 0.0
            }
        
        for result in sparse_results:
            key = f"{result['metadata']['document_id']}_{result['metadata']['chunk_index']}"
            sparse_score = result["metadata"].get("score", 0)
            
            if key in all_results:
                all_results[key]["fusion_score"] += self.sparse_weight * sparse_score
                all_results[key]["sparse_score"] = sparse_score
            else:
                all_results[key] = {
                    **result,
                    "fusion_score": self.sparse_weight * sparse_score,
                    "dense_score": 0.0,
                    "sparse_score": sparse_score
                }
        
        fused = sorted(
            all_results.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )
        
        for result in fused:
            result["metadata"]["fusion_score"] = result["fusion_score"]
            result["metadata"]["dense_score"] = result["dense_score"]
            result["metadata"]["sparse_score"] = result["sparse_score"]
        
        return fused
    


class QueryRewriter:
    """
    Rewrite unclear/vague queries into better search queries
    Uses LLM with LangChain prompt templates
    """
    
    def __init__(self, llm_client: 'HFSpaceLLMClient'):
        self.llm = llm_client
        
        self.rewrite_template = PromptTemplate(
            input_variables=["query"],
            template="""Rewrite this search query to be more specific and clear for document retrieval.

Original query: {query}

Instructions:
- Make it more specific and detailed
- Keep it as a question if possible
- Don't add information not in the original query
- If the query is already clear, return it unchanged
- Return ONLY the rewritten query, nothing else

Rewritten query:"""
        )
        
        logger.info("Initialized QueryRewriter with LangChain PromptTemplate")
    
    def rewrite(self, query: str) -> str:
        """Rewrite query for better retrieval"""
        
        # If query is already clear and specific, don't rewrite
        if len(query.split()) > 5 and "?" in query:
            logger.info("Query seems clear, skipping rewrite")
            return query
        
        try:
            # Format prompt using LangChain template
            prompt = self.rewrite_template.format(query=query)
            
            # Generate rewritten query
            rewritten = self.llm.generate(prompt)
            
            # Clean up the response
            rewritten = rewritten.strip().strip('"').strip("'")
            
            # Basic validation
            if len(rewritten) > 200 or len(rewritten) < 3:
                logger.warning("Invalid rewrite, using original")
                return query
            
            logger.info(f"Rewrote query: '{query}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using original query")
            return query


# ============================================================================
# PHASE 3: HyDE GENERATOR (Using LangChain PromptTemplate)
# ============================================================================

class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE)
    Generate a fake answer, then search using its embedding
    Uses LangChain prompt templates
    """
    
    def __init__(self, llm_client: 'HFSpaceLLMClient', encoder: 'EmbeddingEncoder'):
        self.llm = llm_client
        self.encoder = encoder
        
        self.hyde_template = PromptTemplate(
            input_variables=["question"],
            template="""Generate a detailed, factual answer to this question as if you were answering from a knowledge base.

Question: {question}

Instructions:
- Write a comprehensive paragraph (3-5 sentences)
- Be specific and factual
- Include relevant details and context
- Write as if this is from an authoritative document
- Return ONLY the answer paragraph, nothing else

Answer:"""
        )
        
        logger.info("Initialized HyDEGenerator with LangChain PromptTemplate")
    
    def generate_hypothetical_doc(self, query: str) -> str:
        """Generate a hypothetical answer to the query"""
        
        try:
            prompt = self.hyde_template.format(question=query)
            
            hypothetical_doc = self.llm.generate(prompt)
            
            logger.info(f"Generated hypothetical document ({len(hypothetical_doc)} chars)")
            return hypothetical_doc.strip()
            
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            return query 
    
    def get_hyde_embedding(self, query: str) -> List[float]:
        """Get embedding of hypothetical document instead of query"""
        hypothetical_doc = self.generate_hypothetical_doc(query)
        embedding = self.encoder.encode(hypothetical_doc)
        logger.info("Generated HyDE embedding")
        return embedding


class Neo4jGraphManager:
    """
    Manages Neo4j graph database connection and operations
    Uses LangChain's Neo4j integration
    """
    
    def __init__(self, uri: str, username: str, password: str):
        try:
            logger.info(f"Connecting to Neo4j at {uri}")
            self.graph = Neo4jGraph(
                url=uri,
                username=username,
                password=password
            )
            logger.info("Successfully connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def get_graph(self):
        """Get Neo4j graph instance"""
        return self.graph
    
    def add_graph_documents(self, graph_documents):
        """Add graph documents to Neo4j"""
        if not self.graph:
            logger.error("Neo4j not connected")
            return
        
        try:
            self.graph.add_graph_documents(graph_documents)
            logger.info(f"Added {len(graph_documents)} graph documents to Neo4j")
        except Exception as e:
            logger.error(f"Failed to add graph documents: {e}")
    
    def query_graph(self, cypher_query: str):
        """Execute Cypher query on graph"""
        if not self.graph:
            logger.error("Neo4j not connected")
            return []
        
        try:
            result = self.graph.query(cypher_query)
            return result
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    def get_schema(self):
        """Get graph schema"""
        if not self.graph:
            return "Neo4j not connected"
        
        try:
            return self.graph.schema
        except Exception as e:
            logger.error(f"Failed to get schema: {e}")
            return f"Error: {str(e)}"





class GraphDocumentBuilder:
    """
    Builds graph documents from LangChain documents
    Uses LLMGraphTransformer to extract entities and relationships
    """
    
    def __init__(self, llm_client: 'HFSpaceLLMClient'):
        self.llm_wrapper = LangChainLLMWrapper(llm_client=llm_client)
        
        try:
            logger.info("Initializing LLMGraphTransformer...")
            from langchain_experimental.graph_transformers import LLMGraphTransformer
            
            self.llm_transformer = LLMGraphTransformer(
                llm=self.llm_wrapper,
                allowed_nodes=["Person", "Organization", "Location", "Event", "Concept", "Product"],
                allowed_relationships=["WORKS_AT", "LOCATED_IN", "PART_OF", "RELATED_TO", "CREATED_BY"]
            )
            logger.info("✅ LLMGraphTransformer initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLMGraphTransformer: {e}")
            logger.warning("Graph transformation will be disabled")
            self.llm_transformer = None
    
    def build_graph_documents(self, documents: List[Document]):
        """Convert LangChain documents to graph documents"""
        if not self.llm_transformer:
            logger.warning("LLMGraphTransformer not available - skipping graph conversion")
            return []
        
        try:
            logger.info(f"Converting {len(documents)} documents to graph documents...")
            
            # Process in smaller batches to avoid timeout
            batch_size = 5
            all_graph_docs = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
                
                try:
                    graph_docs = self.llm_transformer.convert_to_graph_documents(batch)
                    all_graph_docs.extend(graph_docs)
                    logger.info(f"✅ Batch {i//batch_size + 1} done: {len(graph_docs)} graph docs")
                except Exception as batch_error:
                    logger.error(f"❌ Batch {i//batch_size + 1} failed: {batch_error}")
                    # Continue with next batch instead of failing completely
                    continue
            
            logger.info(f"✅ Created {len(all_graph_docs)} graph documents total")
            return all_graph_docs
            
        except Exception as e:
            logger.error(f"❌ Failed to build graph documents: {e}")
            return []




class LangChainLLMWrapper(LLM):
    """
    FIXED: Proper LangChain LLM wrapper for HFSpaceLLMClient
    This makes your HuggingFace LLM compatible with LangChain tools
    NOW HANDLES ChatPromptValue CONVERSION
    """
    
    llm_client: Any  # Your HFSpaceLLMClient
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        """Return identifier for LLM type"""
        return "custom_hf_space"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Main method LangChain calls to generate text
        FIXED: Properly handles ChatPromptValue objects
        """
        try:
            if hasattr(prompt, 'to_string'):
                prompt_str = prompt.to_string()
                logger.debug("Converted ChatPromptValue using to_string()")
            elif hasattr(prompt, 'to_messages'):
                messages = prompt.to_messages()
                prompt_str = "\n".join([
                    msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in messages
                ])
                logger.debug("Converted ChatPromptValue using to_messages()")
            else:
                prompt_str = str(prompt)
                logger.debug("Using prompt as-is (string)")
            
            logger.info(f"📤 Calling LLM with prompt (length: {len(prompt_str)} chars)")
            
            response = self.llm_client.generate(prompt_str)
            
            if stop:
                for stop_seq in stop:
                    if stop_seq in response:
                        response = response.split(stop_seq)[0]
                        logger.debug(f"Applied stop sequence: {stop_seq}")
            
            logger.info(f"✅ LLM response received (length: {len(response)} chars)")
            return response
            
        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            logger.error(f"Prompt type: {type(prompt)}")
            logger.error(f"Prompt value: {str(prompt)[:200]}...")
            raise
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Return identifying parameters"""
        return {
            "model_name": "hf_space_llm",
            "type": "custom",
            "base_url": getattr(self.llm_client, 'url', 'unknown')
        }
    
    def __call__(self, prompt: str, **kwargs) -> str:
        """Direct call support"""
        return self._call(prompt, **kwargs)
    
    def predict(self, text: str, **kwargs) -> str:
        """Prediction method"""
        return self._call(text, **kwargs)
    
    def invoke(self, input: str, config: Optional[Any] = None, **kwargs) -> str:
        """Modern LangChain invoke method"""
        return self._call(input, **kwargs)



class GraphCypherRetriever:
    """
    Uses GraphCypherQAChain to query Neo4j and retrieve relevant information
    Combines graph query results with vector search results
    """
    
    def __init__(
        self,
        neo4j_manager: Neo4jGraphManager,
        llm_client: 'HFSpaceLLMClient',
        vectordb: QdrantVectorDB,
        encoder: EmbeddingEncoder,
        graph_weight: float = 0.3
    ):
        self.neo4j_manager = neo4j_manager
        self.llm_wrapper = LangChainLLMWrapper(llm_client=llm_client)
        self.vectordb = vectordb
        self.encoder = encoder
        self.graph_weight = graph_weight
        self.cypher_chain = None
        
        if neo4j_manager.graph:
            try:
                logger.info("Initializing GraphCypherQAChain...")
                from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
                
                self.cypher_chain = GraphCypherQAChain.from_llm(
                    llm=self.llm_wrapper,
                    graph=neo4j_manager.graph,
                    verbose=True,
                    return_intermediate_steps=True,
                    top_k=3
                )
                logger.info(" GraphCypherQAChain initialized successfully")
                
            except Exception as e:
                logger.error(f" Failed to initialize GraphCypherQAChain: {e}")
                logger.warning("Graph queries will be disabled")
                self.cypher_chain = None
        else:
            logger.warning("Neo4j not connected, GraphCypherQAChain disabled")
    
    def query_graph(self, query: str) -> Optional[str]:
        """Query the graph using natural language"""
        if not self.cypher_chain:
            logger.debug("GraphCypherQAChain not available")
            return None
        
        try:
            logger.info(f"🔍 Querying graph: {query}")
            
            # Add timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Graph query timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(30)
            
            try:
                response = self.cypher_chain.invoke({"query": query})
                signal.alarm(0)  # Cancel timeout
                
                # Extract result
                graph_result = response.get("result", "")
                
                if graph_result and len(graph_result.strip()) > 5:
                    logger.info(f"✅ Graph result: {graph_result[:100]}...")
                    return graph_result
                else:
                    logger.info("Graph query returned empty result")
                    return None
                    
            except TimeoutError:
                signal.alarm(0)
                logger.warning("⏰ Graph query timeout")
                return None
            
        except Exception as e:
            logger.error(f"❌ Graph query failed: {e}")
            return None
    
    def search_with_graph(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Enhance vector results with graph information
        """
        graph_answer = self.query_graph(query)
        
        if graph_answer and len(graph_answer.strip()) > 10:
            graph_context = {
                "text": f"📊 Knowledge Graph Insight:\n{graph_answer}",
                "metadata": {
                    "document_id": "neo4j_graph",
                    "source": "Neo4j Knowledge Graph",
                    "chunk_index": 0,
                    "score": 1.0,
                    "retrieval_type": "graph_cypher",
                    "graph_score": 1.0
                }
            }
            
            combined_results = [graph_context] + vector_results
            logger.info("✅ Enhanced results with graph knowledge")
        else:
            combined_results = vector_results
            logger.info("ℹ️  No graph enhancement (using vector results only)")
        
        return combined_results[:top_k]