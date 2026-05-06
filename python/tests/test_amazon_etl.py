import json

from services.amazon_etl import (
    normalize_metadata_record,
    normalize_review_record,
    run_etl,
)


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_normalizers_handle_missing_and_string_fields():
    product = normalize_metadata_record(
        {
            "parent_asin": "B001",
            "title": "USB-C Charger",
            "main_category": "Electronics",
            "price": "$19.99",
            "average_rating": "4.5",
            "rating_number": "1,234",
            "features": None,
        }
    )
    review = normalize_review_record(
        {
            "parent_asin": "B001",
            "rating": "5.0",
            "text": None,
            "helpful_vote": "12",
            "verified_purchase": True,
        }
    )

    assert product["price"] == 19.99
    assert product["average_rating"] == 4.5
    assert product["rating_number"] == 1234
    assert product["features"] == []
    assert review["text"] == ""
    assert review["helpful_vote"] == 12
    assert review["verified_purchase"] is True


def test_run_etl_joins_on_parent_asin_and_respects_limits(tmp_path):
    meta_path = tmp_path / "meta.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    output_dir = tmp_path / "processed"

    write_jsonl(
        meta_path,
        [
            {
                "parent_asin": "B001",
                "title": "Noise Cancelling Headphones",
                "main_category": "Electronics",
                "price": "$99.99",
                "average_rating": 4.6,
                "rating_number": 200,
                "features": ["wireless", "travel"],
                "store": "Acme",
            },
            {
                "parent_asin": "B002",
                "title": "USB-C Hub",
                "main_category": "Electronics",
                "price": "$29.99",
            },
            {
                "parent_asin": "B003",
                "title": "Book",
                "main_category": "Books",
            },
        ],
    )
    write_jsonl(
        reviews_path,
        [
            {"parent_asin": "B001", "user_id": "u1", "rating": 5, "text": "Great for flights"},
            {"parent_asin": "B001", "user_id": "u2", "rating": 4, "text": "Comfortable"},
            {"parent_asin": "B002", "user_id": "u1", "rating": 5, "text": "Useful hub"},
            {"parent_asin": "B003", "user_id": "u3", "rating": 5, "text": "Ignored"},
        ],
    )

    summary = run_etl(
        meta_path=meta_path,
        reviews_path=reviews_path,
        output_dir=output_dir,
        category="Electronics",
        max_products=2,
        max_reviews=3,
    )

    assert summary["product_count"] == 2
    assert summary["review_count"] == 3
    assert summary["document_count"] == 2

    docs = [
        json.loads(line)
        for line in (output_dir / "product_documents.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {doc["parent_asin"] for doc in docs} == {"B001", "B002"}
    assert docs[0]["source_text"]
    profiles = json.loads((output_dir / "mock_user_profiles.json").read_text(encoding="utf-8"))
    assert "u1" in profiles
