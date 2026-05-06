# Interview Guide

## One-Minute Project Pitch

This project is a FastAPI-based shopping assistant that uses Amazon Reviews 2023 Electronics data for semantic product recommendation. An offline ETL pipeline streams product metadata and reviews, builds one searchable document per product, embeds those documents, and indexes them in Milvus. At runtime, the system uses a three-agent Supervisor flow for user profiling, semantic product recall plus LLM reranking, and generated marketing copy/recommendation summaries.

## Talking Points

- **Why RAG:** product recommendations are grounded in real metadata and review highlights instead of only model knowledge.
- **Why streaming ETL:** the Amazon dataset is massive, so v1 samples 1,000 products and 10,000 reviews.
- **Why Milvus:** it supports vector search over product documents and matches the existing project infrastructure.
- **Why fallback embeddings:** local tests and demos can still run when the sentence-transformers model is unavailable.
- **Why keep agents:** profile, retrieval/rerank, and copy/summary generation have separate prompts, failures, and outputs.

## Common Questions

### Q1: What is indexed?

One document per product: title, category, brand/store, price, average rating, rating count, selected features, description, and compact review highlights.

### Q2: How does recommendation work?

The query is embedded, Milvus returns semantically similar products, and the LLM reranks candidates using query intent, user profile, ratings, review counts, semantic score, and review highlights.

### Q3: What happens if Milvus is unavailable?

The recommendation agent falls back to the local mock catalog so the API remains usable for demos and tests.

### Q4: How is correctness evaluated?

The evaluation script compares keyword search and semantic search on curated shopping queries such as “noise cancelling headphones for travel.”

## Files To Know

| File | Why it matters |
| --- | --- |
| `python/services/amazon_etl.py` | Dataset normalization and product document generation |
| `python/services/catalog_index.py` | Milvus indexing and search |
| `python/services/embeddings.py` | Local embeddings and fallback embeddings |
| `python/agents/product_rec_agent.py` | Semantic recall and LLM rerank |
| `python/orchestrator/supervisor.py` | Main runtime flow |
| `python/main.py` | API and catalog utility endpoints |
