# Project Plan

## Scope

The project satisfies the proposal by adding an Amazon Reviews RAG backend to the existing three-agent FastAPI service.

## Milestones

| Milestone | Deliverable |
| --- | --- |
| Dataset subset | Stream and normalize Electronics metadata/reviews |
| ETL artifacts | Products, reviews, searchable documents, mock user profiles |
| Vector index | Milvus collection with product embeddings and review highlights |
| Runtime API | Query-aware recommendations and RAG summary |
| Evaluation | Keyword baseline vs semantic retrieval script |

## Commands

```bash
python -m services.amazon_etl \
  --meta-path ../data/raw/meta_Electronics.jsonl.gz \
  --reviews-path ../data/raw/Electronics.jsonl.gz \
  --output-dir ../data/processed \
  --max-products 1000 \
  --max-reviews 10000
```

```bash
curl -X POST http://localhost:8000/api/v1/catalog/reindex \
  -H "Content-Type: application/json" \
  -d '{"processed_dir": "../data/processed", "limit": 1000, "recreate": true}'
```
