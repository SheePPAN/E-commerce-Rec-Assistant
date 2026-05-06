# Resume Template

## Project Entry

Amazon Reviews RAG Shopping Assistant

- Built a FastAPI recommendation backend using Amazon Reviews 2023 Electronics metadata and reviews.
- Implemented a streaming ETL pipeline that normalizes metadata/reviews, joins by `parent_asin`, creates product-level RAG documents, and generates mock user profiles.
- Indexed product documents in Milvus using `sentence-transformers/all-MiniLM-L6-v2` embeddings for semantic retrieval.
- Updated a multi-agent Supervisor workflow to support query-aware semantic recall, LLM reranking, personalized copy, and grounded recommendation summaries.
- Added catalog status/reindex endpoints plus an evaluation script comparing keyword baseline and semantic retrieval.

Tech stack: Python, FastAPI, LangGraph, LangChain, Milvus, sentence-transformers, Pydantic, Docker Compose.

## Interview Keywords

- Retrieval-Augmented Generation
- Amazon Reviews 2023
- Streaming ETL
- Vector search with Milvus
- Semantic recall
- LLM reranking
- Dataset-grounded recommendation summaries
