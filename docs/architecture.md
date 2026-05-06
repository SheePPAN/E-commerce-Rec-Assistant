# Architecture

The project is now an Amazon Reviews RAG recommendation backend. It combines offline ETL, Milvus semantic search, LLM reranking, and LLM-generated recommendation text.

## Runtime Components

| Component | Responsibility |
| --- | --- |
| FastAPI app | HTTP API, catalog utilities, metrics |
| Supervisor | Main three-agent orchestration |
| LangGraph pipeline | Graph-shaped version of the same workflow |
| UserProfileAgent | Profile generation from context/features |
| ProductRecAgent | Milvus semantic recall and LLM rerank |
| MarketingCopyAgent | Copy generation and recommendation summary |
| AmazonCatalogIndex | Milvus collection management, indexing, and search |
| EmbeddingService | `sentence-transformers/all-MiniLM-L6-v2` embeddings with deterministic fallback |
| Amazon ETL | Streams Amazon metadata/reviews into processed product documents |

## Data Flow

```text
Raw Amazon JSONL/GZ
  -> services.amazon_etl
  -> data/processed/product_documents.jsonl
  -> AmazonCatalogIndex.reindex_from_processed()
  -> Milvus amazon_electronics_products
  -> ProductRecAgent semantic recall
  -> LLM rerank
  -> TopN products
  -> marketing copy + recommendation summary
```

## API Contract

`RecommendationRequest` accepts an optional `query`. Old requests without `query` still work and fall back to profile/mock recommendation behavior.

`Product` includes Amazon review fields:

- `average_rating`
- `rating_number`
- `review_count`
- `review_highlights`
- `semantic_score`
- `source_parent_asin`

`RecommendationResponse` includes `recommendation_summary`, a concise RAG answer grounded in product metadata and review highlights.

## Important Defaults

- Category: Electronics
- Max products for v1 ETL: 1,000
- Max reviews for v1 ETL: 10,000
- Vector collection: `amazon_electronics_products`
- Embedding dimension: 384
- Vector DB: Milvus

## Extension Points

1. Replace sampled ETL with larger category ingestion.
2. Add richer review aggregation, such as pros/cons or aspect summaries.
3. Add query classification before retrieval.
4. Add persistent evaluation metrics for keyword vs semantic retrieval.
