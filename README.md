# RAG-Mastery-Phase4

Phase 4 RAG System — Neo4j Graph-Augmented Retrieval-Augmented Generation (RAG)

This repository implements a modular RAG server (FastAPI) that combines dense embeddings, sparse BM25 retrieval, reranking, query intelligence (rewriting), HyDE, compression/deduplication, and an optional Neo4j knowledge-graph augmentation. The orchestration lives in a RAGPipeline class and the building blocks are implemented in `rag_core.py`.

---

Table of contents
- What this project does
- High-level architecture & logic flow
- Key components and classes
- Tech stack (inferred from imports)
- Runtime configuration (environment variables referenced in code)
- How to run
- HTTP API (endpoints & payloads)
- Notes and caveats

---

What the project does
- Exposes a FastAPI HTTP service (`main.py`) that:
  - Ingests documents (JSON or uploaded files), chunks them, embeds them, stores them in Qdrant, indexes them for BM25, and optionally builds/updates a Neo4j graph.
  - Accepts natural language queries and runs a multi-phase RAG pipeline to produce answers with supporting contexts.
  - Provides a health endpoint to surface status of embeddings, Qdrant, LLM, BM25, reranker, query intelligence, and Neo4j graph.
- Implements a multi-phase retrieval pipeline in `rag_pipeline.py` using components implemented in `rag_core.py`.

---

High-level architecture & logic flow

1. Server entrypoint
   - `main.py` defines a FastAPI app and a `RAGPipeline` instance (`rag = RAGPipeline()`).
   - Endpoints: `POST /ingest`, `POST /upload`, `POST /query`, `GET /health`, `GET /`.

2. Initialization (RAGPipeline.__init__)
   - Phase 1 (Representation & storage)
     - `EmbeddingEncoder` — wraps SentenceTransformer model (embedding model set via `settings.EMBEDDING_MODEL`).
     - `QdrantVectorDB` — Qdrant client for vector storage and similarity search.
     - `HFSpaceLLMClient` — a client wrapper that talks to a Hugging Face Space LLM endpoint (HF Space URL/endpoint/token from settings).
     - `DocumentChunker` — uses LangChain text splitters to chunk large documents.
     - `RAGPrompts` — prompt templates for generating final answers.
   - Phase 2 (Retrieval)
     - `BM25Retriever` — sparse retriever using `rank_bm25`.
     - `CrossEncoderReranker` — cross-encoder reranker using `sentence_transformers.CrossEncoder`.
     - `ContextCompressor` — deduplication/redundancy compression using embeddings and thresholds.
     - `HybridRetriever` — fuses dense (Qdrant) and sparse (BM25) results with configurable weights.
   - Phase 3 (Query intelligence)
     - `QueryRewriter` — optional LLM-based query rewriting.
     - `HyDEGenerator` — optional HyDE (generate hypothetical doc then search using its embedding).
   - Phase 4 (Graph augmentation - optional)
     - `Neo4jGraphManager` — Neo4j connection helper (uses `langchain_neo4j`).
     - `GraphDocumentBuilder` — extracts entities/relations using `LLMGraphTransformer` and builds graph documents.
     - `GraphCypherRetriever` — uses `GraphCypherQAChain` (langchain Neo4j chain) to query the graph and combine results with vector retrieval.

3. Ingestion flow (`RAGPipeline.ingest_documents` and `RAGPipeline.ingest_file`)
   - Input documents are chunked (`DocumentChunker.chunk_document`), converted to LangChain `Document` objects.
   - Batch embeddings are computed via `EmbeddingEncoder.encode_batch`.
   - Embeddings + doc chunks are upserted to Qdrant via `QdrantVectorDB.upsert_documents`.
   - If hybrid search enabled, BM25 indexer (`BM25Retriever.index_documents`) indexes text chunks.
   - If graph RAG enabled, `GraphDocumentBuilder` converts chunks to graph documents and `Neo4jGraphManager` inserts them into Neo4j.

4. Query flow (`RAGPipeline.query`)
   - Optional query rewriting via `QueryRewriter`.
   - Retrieval strategies:
     - If HyDE enabled: generate hypothetical answer, get its embedding, and search Qdrant.
     - Else if hybrid enabled: `HybridRetriever.search` fuses dense and sparse retrieval (with optional reranking/compression).
     - Else: dense retrieval from Qdrant using query embedding.
   - Optional Neo4j Graph augmentation: `GraphCypherRetriever.search_with_graph` enhances vector results using graph QA chain output and combines scores/contexts.
   - Apply compression/deduplication (if enabled).
   - Use LLM (`HFSpaceLLMClient`) and prompt templates (`RAGPrompts`) to generate final answer (with context).
   - Return answer, contexts, retrieval count, response time.

---

Key components and classes (as in code)

- main.py
  - FastAPI app, pydantic models:
    - Document, IngestRequest, IngestResponse, UploadResponse, QueryRequest, QueryResponse, HealthResponse
  - Endpoints: /ingest, /upload, /query, /health, /

- rag_pipeline.py
  - RAGPipeline — orchestrator that wires components and exposes:
    - ingest_documents(documents: List[Dict])
    - ingest_file(file: UploadFile, document_id: str | None)
    - query(query: str)
    - health_check()

- rag_core.py (core building blocks)
  - EmbeddingEncoder — wrapper around SentenceTransformer (embedding model from settings).
  - QdrantVectorDB — wraps QdrantClient for create/search/upsert operations.
  - HFSpaceLLMClient — wrapper that calls HF Space endpoints (LLM).
  - RAGPrompts — prompt templates for QA.
  - DocumentChunker — uses langchain text splitters to create chunks.
  - DocumentLoader — file loaders using PyPDF2, docx2txt, etc.
  - BM25Retriever — BM25 sparse retriever (rank_bm25).
  - CrossEncoderReranker — Cross-encoder reranker using sentence_transformers.CrossEncoder.
  - ContextCompressor — embedding-based dedup/redundancy removal.
  - HybridRetriever — fuses dense + sparse results and optionally reranks/compresses.
  - QueryRewriter — LLM-based query rewriting helper.
  - HyDEGenerator — generates hypothetical doc (HyDE) using a PromptTemplate and returns its embedding.
  - Neo4jGraphManager — manages Neo4j connection and insertion (langchain_neo4j).
  - GraphDocumentBuilder — uses LLMGraphTransformer to extract entities/relations for Neo4j ingestion.
  - GraphCypherRetriever — GraphCypherQAChain wrapper combining graph and vectors.

---

Tech stack (inferred from imports in code)
- Python 3.x
- FastAPI (web server)
- uvicorn (ASGI server)
- langchain_core (LangChain core integrations / prompts / documents)
- langchain_text_splitters
- langchain_neo4j and langchain_experimental.graph_transformers
- sentence-transformers (SentenceTransformer & CrossEncoder)
- rank_bm25 (BM25Okapi)
- qdrant-client (QdrantClient & models)
- sklearn (cosine_similarity)
- PyPDF2 (PDF text extraction)
- docx2txt (DOCX extraction)
- requests (HTTP to HF Space)
- numpy
- (optional) Neo4j database (driver provided/used via langchain_neo4j)
- Additional types used: Pydantic (FastAPI models), typing, logging

---

Runtime configuration (environment variables referenced in code)
The `Settings` dataclass in `config.py` reads many environment variables. The following are referenced and important to set:

Required for core functionality:
- QDRANT_URL — Qdrant endpoint
- QDRANT_API_KEY — Qdrant API key
- HF_SPACE_URL — Base URL of the Hugging Face Space providing the LLM
Optional / toggles and credentials:
- HF_SPACE_ENDPOINT — specific HF Space endpoint (optional)
- HF_TOKEN — HF token (if needed)
- ENABLE_HYBRID_SEARCH (default: True)
- ENABLE_RERANKING (default: True)
- ENABLE_COMPRESSION (default: True)
- ENABLE_QUERY_REWRITING (default: True)
- ENABLE_HYDE (default: False)
- ENABLE_GRAPH_RAG (default: True) — if true, also set Neo4j vars:
  - NEO4J_URI
  - NEO4J_USERNAME (default: neo4j)
  - NEO4J_PASSWORD
Tunable retrieval & indexing:
- EMBEDDING_MODEL (default: sentence-transformers/all-MiniLM-L6-v2)
- EMBEDDING_DIM (default: 384)
- CHUNK_SIZE (default: 512)
- CHUNK_OVERLAP (default: 50)
- TOP_K, HYBRID_TOP_K, FINAL_TOP_K, GRAPH_TOP_K
- DENSE_WEIGHT, SPARSE_WEIGHT
- BM25_K1, BM25_B
- RERANKER_MODEL (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
- DEDUP_THRESHOLD, REDUNDANCY_THRESHOLD
- MAX_UPLOAD_SIZE

The code also performs validation and logs configuration on startup. If `ENABLE_GRAPH_RAG` is true but NEO4J_URI or password missing, the code warns and may disable or break graph features.

---

How to run (based strictly on code)

1. Install dependencies (packages inferred from imports; there is no lockfile in this repo — the following list is the minimum inferred set):
   - fastapi
   - uvicorn
   - langchain-core (or langchain_core as used in imports)
   - langchain-text-splitters
   - langchain-neo4j
   - langchain-experimental (graph_transformers)
   - sentence-transformers
   - rank_bm25
   - qdrant-client
   - scikit-learn
   - numpy
   - PyPDF2
   - docx2txt
   - requests
   - pydantic

   Example (pip):
   pip install fastapi uvicorn sentence-transformers rank_bm25 qdrant-client scikit-learn numpy PyPDF2 docx2txt requests

   Note: `langchain_core`, `langchain_text_splitters`, `langchain_neo4j` and `langchain_experimental.graph_transformers` are imported in the code; make sure to install the correct langchain packages that provide these modules compatible with your environment.

2. Set environment variables required by `config.Settings`. At minimum set:
   - QDRANT_URL, QDRANT_API_KEY, HF_SPACE_URL

   If using Neo4j graph features:
   - ENABLE_GRAPH_RAG=true
   - NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

   Example (Linux/macOS):
   export QDRANT_URL="https://your-qdrant.example"
   export QDRANT_API_KEY="your_api_key"
   export HF_SPACE_URL="https://hf.space/run/your-llm-space"
   # Optional:
   export NEO4J_URI="bolt://neo4j:7687"
   export NEO4J_PASSWORD="your_neo4j_password"

3. Start the server
   - Either:
     python main.py
     (The `if __name__ == "__main__":` block starts uvicorn with host 0.0.0.0:8000)
   - Or run uvicorn directly:
     uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info

4. Use the API
   - Open interactive docs at: GET /docs (FastAPI auto-generated docs)
   - Health check:
     GET http://localhost:8000/health
   - Ingest documents (example JSON body):
     POST http://localhost:8000/ingest
     Content-Type: application/json
     Body:
     {
       "documents": [
         {
           "document_id": "doc-1",
           "source": "mydoc",
           "text": "Full text of the document...",
           "metadata": {"author": "alice"}
         }
       ]
     }
   - Upload file:
     POST http://localhost:8000/upload
     Content-Type: multipart/form-data
     - file: (attach file)
     - document_id: optional form field
   - Query:
     POST http://localhost:8000/query
     Content-Type: application/json
     Body:
     {
       "query": "How do I configure the Qdrant collection?"
     }

Response models and shapes are defined with Pydantic in `main.py`:
- QueryResponse contains: query (string), answer (string), contexts (list of context dicts), retrieval_count (int), response_time_ms (int)
- IngestResponse and UploadResponse contain ingestion summary fields.

---

Notes, limitations and caveats (from the code)
- Many features are optional and controlled by environment flags (`ENABLE_HYBRID_SEARCH`, `ENABLE_RERANKING`, `ENABLE_COMPRESSION`, `ENABLE_QUERY_REWRITING`, `ENABLE_HYDE`, `ENABLE_GRAPH_RAG`). If required credentials or endpoints are missing the code will warn and may disable features.
- The HF LLM integration is implemented as a wrapper `HFSpaceLLMClient` that calls a Hugging Face Space endpoint — ensure the Space endpoint and token (if needed) are reachable and compatible.
- Neo4j graph features require `langchain_neo4j` and a running Neo4j instance reachable with credentials. If Neo4j is not available, graph-related functionality will be disabled or return warnings.
- The exact versions of langchain / langchain-core modules expected by the code are not specified in this repository. You may need to align package versions to satisfy imports like `langchain_core` and `langchain_neo4j`.


- Create example curl commands and a Postman collection for ingest/query flows.
- Draft a CONTRIBUTING.md or developer guide explaining how to add new retrievers or LLM backends.
