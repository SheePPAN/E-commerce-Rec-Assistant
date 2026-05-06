from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.catalog_index import AmazonCatalogIndex
from services.embeddings import EmbeddingService


DEFAULT_QUERIES = [
    "noise cancelling headphones for travel",
    "portable bluetooth speaker with strong battery",
    "affordable wireless charger for phone",
    "lightweight laptop accessory for students",
    "comfortable gaming headset with good microphone",
    "usb c hub for macbook and monitor",
    "high quality monitor for home office",
    "compact camera for vlogging",
    "durable phone case with screen protection",
    "fast external storage for backups",
]


def load_documents(processed_dir: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    path = Path(processed_dir) / "product_documents.jsonl"
    docs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            docs.append(json.loads(line))
            if limit and len(docs) >= limit:
                break
    return docs


def keyword_search(query: str, documents: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    terms = [term.lower() for term in query.split() if term.strip()]
    scored = []
    for doc in documents:
        text = (doc.get("source_text") or "").lower()
        score = sum(text.count(term) for term in terms)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


def cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return numerator / (left_norm * right_norm)


def local_semantic_search(
    query: str,
    documents: list[dict[str, Any]],
    embeddings: EmbeddingService,
    limit: int,
) -> list[dict[str, Any]]:
    query_vector = embeddings.embed_query(query)
    doc_vectors = embeddings.embed_texts(doc.get("source_text") or doc.get("title", "") for doc in documents)
    scored = sorted(
        zip(documents, doc_vectors),
        key=lambda item: cosine(query_vector, item[1]),
        reverse=True,
    )
    return [doc for doc, _ in scored[:limit]]


def run_evaluation(processed_dir: str | Path, use_milvus: bool, limit: int = 5) -> list[dict[str, Any]]:
    documents = load_documents(processed_dir)
    embeddings = EmbeddingService(force_fallback=not use_milvus)
    catalog = AmazonCatalogIndex(embedding_service=embeddings)

    rows = []
    for query in DEFAULT_QUERIES:
        keyword_results = keyword_search(query, documents, limit)
        if use_milvus:
            semantic_products = catalog.search(query, limit=limit)
            semantic_titles = [product.name for product in semantic_products]
        else:
            semantic_titles = [
                doc.get("title", "") for doc in local_semantic_search(query, documents, embeddings, limit)
            ]
        rows.append(
            {
                "query": query,
                "keyword_top_titles": [doc.get("title", "") for doc in keyword_results],
                "semantic_top_titles": semantic_titles,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare keyword and semantic retrieval.")
    parser.add_argument("--processed-dir", default="../data/processed")
    parser.add_argument("--use-milvus", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    results = run_evaluation(args.processed_dir, args.use_milvus, args.limit)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
