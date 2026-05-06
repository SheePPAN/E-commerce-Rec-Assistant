# Amazon Reviews RAG Shopping Assistant

This project is a FastAPI-only backend for an intelligent shopping decision assistant. It uses a small Electronics subset from Amazon Reviews 2023, builds product documents from metadata and review snippets, indexes them in Milvus, and recommends products for natural-language shopping queries.

The current runtime still uses a three-agent Supervisor flow:

- `UserProfileAgent` builds a user profile from request context or a feature store.
- `ProductRecAgent` recalls products through Amazon review semantic search and reranks them with an LLM.
- `MarketingCopyAgent` generates product copy plus a dataset-grounded recommendation summary.

Items are unlimited. There is no inventory, stock, or purchase-limit logic.

## Architecture

```text
Amazon Reviews 2023 Electronics files
  |
  v
Offline ETL
  - products.jsonl
  - reviews.jsonl
  - product_documents.jsonl
  - mock_user_profiles.json
  |
  v
Milvus product vector index
  |
  v
FastAPI /api/v1/recommend
  |
  v
Profile + semantic recall -> LLM rerank -> TopN -> copy + summary
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/recommend` | Supervisor-based RAG recommendation |
| `POST` | `/api/v1/recommend/graph` | LangGraph-based RAG recommendation |
| `GET` | `/api/v1/catalog/status` | Milvus catalog status |
| `POST` | `/api/v1/catalog/reindex` | Local/dev reindex from processed documents |
| `GET` | `/api/v1/metrics` | In-memory agent metrics |

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "query": "noise cancelling headphones for travel",
    "num_items": 5,
    "context": {
      "recent_views": ["headphones", "bluetooth speaker"],
      "avg_order_amount": 120
    }
  }'
```

## Dataset Setup

Download the Amazon Reviews 2023 Electronics review and metadata JSONL files from the UCSD dataset site: <https://amazon-reviews-2023.github.io/main.html>.

Place raw files locally, for example:

```text
data/raw/Electronics.jsonl.gz
data/raw/meta_Electronics.jsonl.gz
```

Raw and processed data are ignored by git.

## ETL

Create a small demo subset:

```bash
cd python
python -m services.amazon_etl \
  --meta-path ../data/raw/meta_Electronics.jsonl.gz \
  --reviews-path ../data/raw/Electronics.jsonl.gz \
  --output-dir ../data/processed \
  --category Electronics \
  --max-products 1000 \
  --max-reviews 10000
```

Generated artifacts:

- `data/processed/products.jsonl`
- `data/processed/reviews.jsonl`
- `data/processed/product_documents.jsonl`
- `data/processed/mock_user_profiles.json`
- `data/processed/etl_summary.json`

## Index Into Milvus

Start services:

```bash
docker-compose up -d
```

Run the API, then index processed documents:

```bash
curl -X POST http://localhost:8000/api/v1/catalog/reindex \
  -H "Content-Type: application/json" \
  -d '{"processed_dir": "../data/processed", "limit": 1000, "recreate": true}'
```

Check status:

```bash
curl http://localhost:8000/api/v1/catalog/status
```

## Evaluation

Compare keyword and semantic retrieval over curated shopping queries:

```bash
cd python
python evaluation/evaluate_retrieval.py --processed-dir ../data/processed
```

Use Milvus-backed semantic search when the index is running:

```bash
python evaluation/evaluate_retrieval.py --processed-dir ../data/processed --use-milvus
```

## Project Layout

```text
python/
├── main.py
├── agents/
│   ├── base_agent.py
│   ├── marketing_copy_agent.py
│   ├── product_rec_agent.py
│   └── user_profile_agent.py
├── evaluation/
│   └── evaluate_retrieval.py
├── models/
│   └── schemas.py
├── orchestrator/
│   ├── graph.py
│   └── supervisor.py
└── services/
    ├── amazon_etl.py
    ├── catalog_index.py
    ├── embeddings.py
    ├── feature_store.py
    └── metrics.py
```
