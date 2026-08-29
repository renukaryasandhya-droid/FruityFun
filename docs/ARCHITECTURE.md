# Fruity Fun Architecture

```mermaid
flowchart TB
    USER([Child, parent, or teacher])
    UI[Streamlit chat interface]

    subgraph CONFIG[Configuration]
        ENV[.env]
        SETTINGS[Pydantic Settings]
        ENV --> SETTINGS
    end

    subgraph INGEST[Offline corpus ingestion]
        PDF[Fruit PDF corpus]
        PARSE[PyMuPDF extraction]
        CHUNK[Clean and page-aware chunking]
        IDS[Deterministic SHA-256 record IDs]
        JSON[(Local JSON corpus)]
        EMBED_DOCS[OpenAI document embeddings]
        PINECONE[(Pinecone dense index<br/>fruit-corpus-v1 namespace)]

        PDF --> PARSE --> CHUNK --> IDS
        IDS --> JSON
        IDS --> EMBED_DOCS --> PINECONE
    end

    subgraph GRAPH[LangGraph request workflow]
        START([Request])
        SAFETY{Kid-safe request?}
        REDIRECT[Replace with cheerful fruit scene]
        RETRIEVE[Hybrid retriever]
        DENSE[Dense semantic search]
        SPARSE[BM25 keyword search]
        FUSION[Reciprocal-rank fusion]
        RERANK[LLM reranking]
        CONFIDENCE{Confidence at least 0.65?}
        GROUNDED[Grounded answer with citations]
        IMAGINATION[Insufficient-evidence response]
        IMAGE_PROMPT[Safe picture prompt]
        IMAGE_API[OpenAI image generation]
        IMAGE_FILE[(Generated PNG)]
        RESULT[Answer, image, sources, and confidence]

        START --> SAFETY
        SAFETY -- Yes --> RETRIEVE
        SAFETY -- No --> REDIRECT --> IMAGE_PROMPT
        RETRIEVE --> DENSE --> FUSION
        RETRIEVE --> SPARSE --> FUSION
        FUSION --> RERANK --> CONFIDENCE
        CONFIDENCE -- Yes --> GROUNDED
        CONFIDENCE -- No --> IMAGINATION
        GROUNDED --> IMAGE_PROMPT
        IMAGINATION --> IMAGE_PROMPT
        IMAGE_PROMPT --> IMAGE_API --> IMAGE_FILE
        GROUNDED --> RESULT
        IMAGINATION --> RESULT
        IMAGE_FILE --> RESULT
    end

    USER --> UI --> START
    RESULT --> UI --> USER

    JSON --> SPARSE
    JSON -. source and page metadata .-> RESULT
    PINECONE --> DENSE

    SETTINGS -. models, index, namespace, thresholds .-> INGEST
    SETTINGS -. models, index, namespace, thresholds .-> GRAPH

    OPENAI_LLM[(OpenAI LLM)]
    RERANK --> OPENAI_LLM
    OPENAI_LLM --> RERANK
    OPENAI_LLM --> GROUNDED
    OPENAI_LLM --> IMAGINATION
```

## Main execution paths

- **Ingestion:** `scripts/ingest.py` extracts PDFs, creates deterministic chunks, writes the local JSON corpus, generates embeddings, and upserts records into Pinecone.
- **Retrieval:** `HybridRetriever` queries Pinecone and BM25, combines their rankings with RRF, and calculates evidence confidence after reranking.
- **Orchestration:** `FruityFunAgent` uses LangGraph to move each request through safety, retrieval, reranking, answer composition, and image generation.
- **Presentation:** `app.py` keeps chat history and displays the answer, generated picture, citations, retrieval channels, and confidence.

## Failure behavior

```mermaid
flowchart LR
    PINECONE_FAIL[Pinecone unavailable] --> BM25_ONLY[Continue with local BM25]
    RERANK_FAIL[Reranker unavailable or malformed] --> RRF_ONLY[Keep deterministic RRF order]
    LOW[Low retrieval confidence] --> NO_FACTS[Disclose insufficient evidence]
    IMAGE_FAIL[Image generation unavailable] --> TEXT_ONLY[Keep the text answer visible]
```
