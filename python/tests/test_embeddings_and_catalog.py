from services.catalog_index import AmazonCatalogIndex
from services.embeddings import EmbeddingService


def test_fallback_embedding_dimension_matches_configuration():
    embeddings = EmbeddingService(dimension=384, force_fallback=True)
    vector = embeddings.embed_query("noise cancelling headphones")
    assert len(vector) == 384
    assert any(value != 0 for value in vector)


def test_catalog_document_row_contains_expected_fields():
    embeddings = EmbeddingService(dimension=384, force_fallback=True)
    catalog = AmazonCatalogIndex(embedding_service=embeddings)
    vector = embeddings.embed_query("portable speaker")
    row = catalog._document_to_row(
        {
            "product_id": "B001",
            "parent_asin": "B001",
            "title": "Portable Speaker",
            "category": "Electronics",
            "brand": "Acme",
            "price": 39.99,
            "average_rating": 4.2,
            "rating_number": 100,
            "review_count": 5,
            "features": ["bluetooth"],
            "description": "Small speaker",
            "image_url": "https://example.com/img.jpg",
            "review_highlights": ["Battery lasts long"],
            "source_text": "Portable bluetooth speaker with long battery",
        },
        vector,
    )

    assert row["product_id"] == "B001"
    assert row["average_rating"] == 4.2
    assert row["review_count"] == 5
    assert len(row["embedding"]) == 384
