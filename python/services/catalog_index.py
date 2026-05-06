from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import get_settings
from models.schemas import CatalogStatus, Product

from .embeddings import EmbeddingService


class AmazonCatalogIndex:
    """Milvus-backed product search over processed Amazon review documents."""

    def __init__(
        self,
        collection_name: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        settings = get_settings()
        self.settings = settings
        self.collection_name = collection_name or settings.milvus_collection
        self.embedding_service = embedding_service or EmbeddingService(
            model_name=settings.embedding_model,
            dimension=settings.embedding_dimension,
        )
        self._collection = None
        self._last_error: str | None = None

    def status(self) -> CatalogStatus:
        try:
            collection = self._get_collection(load=False)
            entity_count = int(getattr(collection, "num_entities", 0))
            loaded = entity_count > 0
            return CatalogStatus(
                collection_name=self.collection_name,
                embedding_model=self.embedding_service.model_name,
                embedding_dimension=self.settings.embedding_dimension,
                available=True,
                loaded=loaded,
                entity_count=entity_count,
                error=self._last_error,
            )
        except Exception as exc:
            self._last_error = str(exc)
            return CatalogStatus(
                collection_name=self.collection_name,
                embedding_model=self.embedding_service.model_name,
                embedding_dimension=self.settings.embedding_dimension,
                available=False,
                loaded=False,
                entity_count=0,
                error=str(exc),
            )

    def search(self, query: str, limit: int = 10) -> list[Product]:
        if not query.strip():
            return []
        try:
            collection = self._get_collection(load=True)
            vector = self.embedding_service.embed_query(query)
            results = collection.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=limit,
                output_fields=[
                    "product_id",
                    "parent_asin",
                    "title",
                    "category",
                    "brand",
                    "price",
                    "average_rating",
                    "rating_number",
                    "review_count",
                    "features",
                    "description",
                    "image_url",
                    "review_highlights",
                    "source_text",
                ],
            )
            hits = results[0] if results else []
            return [self._hit_to_product(hit) for hit in hits]
        except Exception as exc:
            self._last_error = str(exc)
            return []

    def reindex_from_processed(
        self, processed_dir: str | Path, limit: int | None = None, recreate: bool = False
    ) -> dict[str, Any]:
        documents = self.load_documents(processed_dir, limit)
        inserted = self.index_documents(documents, recreate=recreate)
        return {
            "collection_name": self.collection_name,
            "processed_dir": str(processed_dir),
            "document_count": len(documents),
            "inserted": inserted,
        }

    def load_documents(self, processed_dir: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
        path = Path(processed_dir) / "product_documents.jsonl"
        documents = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                documents.append(json.loads(line))
                if limit and len(documents) >= limit:
                    break
        return documents

    def index_documents(self, documents: list[dict[str, Any]], recreate: bool = False) -> int:
        if not documents:
            return 0
        collection = self._ensure_collection(recreate=recreate)
        vectors = self.embedding_service.embed_texts(
            [doc.get("source_text") or doc.get("title", "") for doc in documents]
        )
        rows = [self._document_to_row(doc, vector) for doc, vector in zip(documents, vectors)]
        collection.insert(rows)
        collection.flush()
        self._create_index(collection)
        collection.load()
        return len(rows)

    def _get_collection(self, load: bool):
        if self._collection is None:
            from pymilvus import Collection, connections, utility

            connections.connect(
                alias="default",
                host=self.settings.milvus_host,
                port=str(self.settings.milvus_port),
            )
            if not utility.has_collection(self.collection_name):
                raise RuntimeError(f"Milvus collection '{self.collection_name}' does not exist")
            self._collection = Collection(self.collection_name)
        if load:
            self._collection.load()
        return self._collection

    def _ensure_collection(self, recreate: bool):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=str(self.settings.milvus_port),
        )
        if utility.has_collection(self.collection_name):
            if recreate:
                utility.drop_collection(self.collection_name)
            else:
                self._collection = Collection(self.collection_name)
                return self._collection

        fields = [
            FieldSchema("product_id", DataType.VARCHAR, is_primary=True, max_length=128),
            FieldSchema("parent_asin", DataType.VARCHAR, max_length=128),
            FieldSchema("title", DataType.VARCHAR, max_length=1024),
            FieldSchema("category", DataType.VARCHAR, max_length=256),
            FieldSchema("brand", DataType.VARCHAR, max_length=256),
            FieldSchema("price", DataType.FLOAT),
            FieldSchema("average_rating", DataType.FLOAT),
            FieldSchema("rating_number", DataType.INT64),
            FieldSchema("review_count", DataType.INT64),
            FieldSchema("features", DataType.VARCHAR, max_length=4096),
            FieldSchema("description", DataType.VARCHAR, max_length=4096),
            FieldSchema("image_url", DataType.VARCHAR, max_length=1024),
            FieldSchema("review_highlights", DataType.VARCHAR, max_length=4096),
            FieldSchema("source_text", DataType.VARCHAR, max_length=12000),
            FieldSchema(
                "embedding",
                DataType.FLOAT_VECTOR,
                dim=self.settings.embedding_dimension,
            ),
        ]
        schema = CollectionSchema(fields, description="Amazon Electronics product review RAG index")
        self._collection = Collection(self.collection_name, schema=schema)
        return self._collection

    def _create_index(self, collection) -> None:
        try:
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
        except Exception:
            pass

    def _document_to_row(self, doc: dict[str, Any], vector: list[float]) -> dict[str, Any]:
        return {
            "product_id": str(doc.get("product_id") or doc.get("parent_asin")),
            "parent_asin": str(doc.get("parent_asin") or doc.get("product_id")),
            "title": self._clip(doc.get("title"), 1024),
            "category": self._clip(doc.get("category"), 256),
            "brand": self._clip(doc.get("brand"), 256),
            "price": float(doc.get("price") or 0.0),
            "average_rating": float(doc.get("average_rating") or 0.0),
            "rating_number": int(doc.get("rating_number") or 0),
            "review_count": int(doc.get("review_count") or 0),
            "features": self._clip(json.dumps(doc.get("features", []), ensure_ascii=False), 4096),
            "description": self._clip(doc.get("description"), 4096),
            "image_url": self._clip(doc.get("image_url"), 1024),
            "review_highlights": self._clip(
                json.dumps(doc.get("review_highlights", []), ensure_ascii=False), 4096
            ),
            "source_text": self._clip(doc.get("source_text"), 12000),
            "embedding": vector,
        }

    def _hit_to_product(self, hit) -> Product:
        entity = getattr(hit, "entity", None)
        get = entity.get if entity is not None else hit.get
        highlights_raw = get("review_highlights") or "[]"
        try:
            highlights = json.loads(highlights_raw)
        except json.JSONDecodeError:
            highlights = []
        features_raw = get("features") or "[]"
        try:
            tags = json.loads(features_raw)
        except json.JSONDecodeError:
            tags = []
        distance = float(getattr(hit, "distance", 0.0))
        score = max(0.0, min(1.0, distance))
        return Product(
            product_id=str(get("product_id") or get("parent_asin")),
            source_parent_asin=str(get("parent_asin") or get("product_id")),
            name=str(get("title") or ""),
            category=str(get("category") or ""),
            price=float(get("price") or 0.0),
            description=str(get("description") or ""),
            brand=str(get("brand") or ""),
            tags=[str(tag) for tag in tags[:10]],
            image_url=str(get("image_url") or ""),
            average_rating=float(get("average_rating") or 0.0),
            rating_number=int(get("rating_number") or 0),
            review_count=int(get("review_count") or 0),
            review_highlights=[str(item) for item in highlights[:5]],
            semantic_score=score,
            score=score,
        )

    def _clip(self, value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        return text[:limit]
