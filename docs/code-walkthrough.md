# Code Walkthrough

## 1. ETL

`python/services/amazon_etl.py` streams raw Amazon metadata and review JSONL/GZ files. It normalizes product metadata, normalizes reviews, joins them by `parent_asin`, builds one searchable document per product, and writes processed artifacts.

Run it with:

```bash
python -m services.amazon_etl \
  --meta-path ../data/raw/meta_Electronics.jsonl.gz \
  --reviews-path ../data/raw/Electronics.jsonl.gz \
  --output-dir ../data/processed \
  --max-products 1000 \
  --max-reviews 10000
```

## 2. Embeddings

`python/services/embeddings.py` wraps `sentence-transformers/all-MiniLM-L6-v2`. If the model is unavailable, it uses a deterministic hash embedding fallback so tests and local imports still work.

## 3. Milvus Catalog Index

`python/services/catalog_index.py` owns:

- Milvus collection creation,
- document embedding,
- vector insertion,
- semantic search,
- conversion of Milvus hits into `Product` models.

The collection name defaults to `amazon_electronics_products`.

## 4. Recommendation Agent

`python/agents/product_rec_agent.py` now uses semantic search when `RecommendationRequest.query` is present. If Milvus is unavailable or the query is omitted, it falls back to the local mock product list.

Reranking considers:

- user query,
- user profile,
- product metadata,
- ratings/review counts,
- review highlights,
- semantic score.

## 5. Supervisor Flow

`python/orchestrator/supervisor.py` keeps the three-agent flow:

```text
profile + recall in parallel
  -> rerank
  -> TopN
  -> marketing copy
  -> recommendation summary
```

The graph version in `python/orchestrator/graph.py` mirrors the same flow.

## 6. API Layer

`python/main.py` exposes recommendation endpoints plus catalog utilities:

- `GET /api/v1/catalog/status`
- `POST /api/v1/catalog/reindex`

The reindex endpoint is intended for local/demo use after ETL has created `product_documents.jsonl`.
