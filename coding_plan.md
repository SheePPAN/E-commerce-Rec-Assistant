# Plan: Satisfy `proposal.md` With an Amazon Reviews RAG Recommendation Backend

## Summary

Build a FastAPI-only v1 that uses a small Electronics subset from Amazon Reviews 2023, indexes product/review documents into Milvus, and answers semantic shopping queries with dataset-grounded product recommendations.

Chosen defaults:
- App shape: **FastAPI only**
- Vector DB: **Milvus**
- Dataset scope: **1,000 Electronics products + up to 10,000 reviews**
- Embeddings: local `sentence-transformers/all-MiniLM-L6-v2` to keep cost low
- Dataset basis: Amazon Reviews 2023 Electronics has separate review/meta JSONL files and is very large, so streaming/sampling is required ([Amazon Reviews 2023 docs](https://amazon-reviews-2023.github.io/main.html)).

## Key Changes

### Data + ETL

- Add an offline ETL pipeline that reads local Electronics review/meta JSONL files, streams records, and writes processed artifacts under `data/processed/`.
- Normalize metadata into product records using `parent_asin`, `title`, `main_category`, `price`, `average_rating`, `rating_number`, `features`, `description`, `store`, `categories`, `details`, and image URL.
- Normalize reviews using `parent_asin`, `user_id`, `rating`, `title`, `text`, `timestamp`, `verified_purchase`, and helpful-vote fields.
- Build one searchable document per product by combining product metadata plus selected review snippets.
- Generate mock user profiles from retained reviews: preferred categories, liked products, rating distribution, recent review text, and inferred tags.

### Retrieval + RAG Backend

- Add a Milvus indexing service with a collection like `amazon_electronics_products`.
- Store vectors with fields: `product_id`, `parent_asin`, `title`, `category`, `brand/store`, `price`, `average_rating`, `rating_number`, `review_count`, `source_text`, and compact review highlights.
- Update `ProductRecAgent` so `_recall()` uses query embedding + Milvus semantic search instead of hardcoded mock products.
- Update reranking prompt to consider `query`, `user_profile`, product metadata, review highlights, rating count, and semantic score.
- Keep TopN direct selection; no inventory, no stock, no purchase limits.

### API + Models

- Extend `RecommendationRequest` with optional `query: str`.
- Extend `Product` with Amazon dataset fields: `average_rating`, `rating_number`, `review_count`, `review_highlights`, `semantic_score`, and `source_parent_asin`.
- Extend `RecommendationResponse` with `recommendation_summary: str` for the RAG-generated shopping answer.
- Keep existing endpoints:
  - `POST /api/v1/recommend`
  - `POST /api/v1/recommend/graph`
  - `GET /api/v1/metrics`
- Add utility endpoints only if needed for demo/debug:
  - `GET /api/v1/catalog/status` to report whether the Milvus collection is loaded.
  - `POST /api/v1/catalog/reindex` only for local/dev use.

### Documentation + Evaluation

- Update README/docs to describe the Amazon Electronics ETL, Milvus index, semantic search, and RAG flow.
- Add an evaluation script comparing keyword baseline vs semantic retrieval for 10-20 curated shopping queries.
- Document how to download the dataset manually and run ETL with:
  - `--max-products 1000`
  - `--max-reviews 10000`
  - `--category Electronics`

## Test Plan

- ETL tests:
  - Metadata and review JSONL parsing handles missing/null fields.
  - Product/review joins use `parent_asin`.
  - Sampling limits produce no more than 1,000 products and 10,000 reviews.
- Model tests:
  - New `RecommendationRequest` accepts old requests without `query`.
  - New `Product` fields serialize correctly.
- Retrieval tests:
  - Embedding dimensions match Milvus collection schema.
  - Indexing inserts expected product count.
  - Semantic search returns products with `semantic_score` and review highlights.
- API tests:
  - `/api/v1/recommend` works with a natural-language query.
  - Existing context-based recommendation still works when `query` is omitted.
- Evaluation:
  - Run curated queries such as “noise cancelling headphones for travel” and compare semantic results against keyword baseline.

## Assumptions

- Raw Amazon dataset files are not committed to git.
- The implementer will add `data/raw/` and `data/processed/` to `.gitignore`.
- Milvus remains the only v1 vector database.
- Streamlit is intentionally out of scope because the chosen app shape is FastAPI only.
- The first version prioritizes a reliable demo over full-scale ingestion.
