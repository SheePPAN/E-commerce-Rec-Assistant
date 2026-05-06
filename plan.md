# Project Plan

## Current Scope

Build a FastAPI-only Amazon Reviews RAG backend for shopping recommendations.

1. Stream a small Electronics subset from Amazon Reviews 2023.
2. Build product documents from metadata plus review snippets.
3. Index documents into Milvus with local embeddings.
4. Use semantic recall, LLM reranking, and generated recommendation summaries.

## Delivery Phases

| Phase | Goal | Output |
| --- | --- | --- |
| 1 | ETL | `products.jsonl`, `reviews.jsonl`, `product_documents.jsonl`, mock profiles |
| 2 | Indexing | Milvus collection `amazon_electronics_products` |
| 3 | Runtime | Query-aware recommendation endpoint and catalog status/reindex endpoints |
| 4 | Evaluation | Keyword vs semantic retrieval comparison script |
| 5 | Quality | ETL, schema, embedding, and catalog tests |

## Target Flow

```text
Raw Amazon files
  -> ETL
  -> processed product documents
  -> Milvus vector index
  -> semantic recall
  -> LLM rerank
  -> TopN products
  -> copy + recommendation summary
```

## Defaults

- Category: Electronics
- Max products: 1,000
- Max reviews: 10,000
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Milvus
