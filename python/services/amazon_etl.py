from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def open_jsonl(path: str | Path):
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8")
    return file_path.open("r", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with open_jsonl(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def normalize_price(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    text = str(value)
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    return float(match.group()) if match else 0.0


def normalize_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", str(value).replace(",", ""))
    return int(match.group()) if match else 0


def normalize_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+", str(value).replace(",", ""))
    return float(match.group()) if match else 0.0


def normalize_categories(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, list):
                result.extend(str(part).strip() for part in item if str(part).strip())
            elif item is not None:
                text = str(item).strip()
                if text:
                    result.append(text)
        return result
    return [str(value).strip()]


def extract_image_url(value: Any) -> str:
    if not value:
        return ""
    images = value if isinstance(value, list) else [value]
    for image in images:
        if isinstance(image, dict):
            for key in ("large", "hi_res", "thumb", "main", "variant"):
                url = image.get(key)
                if isinstance(url, str) and url.startswith("http"):
                    return url
        elif isinstance(image, str) and image.startswith("http"):
            return image
    return ""


def category_matches(product: dict[str, Any], category: str | None) -> bool:
    if not category:
        return True
    needle = category.lower()
    haystack = " ".join(
        [
            product.get("main_category", ""),
            product.get("category", ""),
            " ".join(product.get("categories", [])),
        ]
    ).lower()
    return needle in haystack


def normalize_metadata_record(raw: dict[str, Any]) -> dict[str, Any]:
    categories = normalize_categories(raw.get("categories"))
    main_category = normalize_text(raw.get("main_category")) or (
        categories[0] if categories else ""
    )
    title = normalize_text(raw.get("title"))
    parent_asin = normalize_text(raw.get("parent_asin") or raw.get("asin"))
    store = normalize_text(raw.get("store"))
    features = normalize_categories(raw.get("features"))
    description = normalize_text(raw.get("description"))
    details = raw.get("details") if isinstance(raw.get("details"), dict) else {}

    return {
        "product_id": parent_asin,
        "parent_asin": parent_asin,
        "title": title,
        "main_category": main_category,
        "category": main_category,
        "price": normalize_price(raw.get("price")),
        "average_rating": normalize_float(raw.get("average_rating")),
        "rating_number": normalize_int(raw.get("rating_number")),
        "features": features,
        "description": description,
        "store": store,
        "brand": store,
        "categories": categories,
        "details": details,
        "image_url": extract_image_url(raw.get("images")),
    }


def normalize_review_record(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": normalize_text(raw.get("parent_asin") or raw.get("asin")),
        "user_id": normalize_text(raw.get("user_id")),
        "rating": normalize_float(raw.get("rating")),
        "title": normalize_text(raw.get("title")),
        "text": normalize_text(raw.get("text")),
        "timestamp": normalize_int(raw.get("timestamp")),
        "verified_purchase": bool(raw.get("verified_purchase", False)),
        "helpful_vote": normalize_int(raw.get("helpful_vote")),
    }


def product_source_text(product: dict[str, Any], reviews: list[dict[str, Any]]) -> str:
    review_lines = []
    for review in reviews[:5]:
        line = " ".join(
            part
            for part in [
                f"rating {review.get('rating', 0):.1f}",
                review.get("title", ""),
                review.get("text", ""),
            ]
            if part
        )
        if line:
            review_lines.append(line[:700])

    parts = [
        product.get("title", ""),
        product.get("main_category", ""),
        product.get("store", ""),
        " ".join(product.get("categories", [])),
        " ".join(product.get("features", [])),
        product.get("description", ""),
        " ".join(review_lines),
    ]
    return "\n".join(part for part in parts if part).strip()


def review_highlights(reviews: list[dict[str, Any]], limit: int = 5) -> list[str]:
    sorted_reviews = sorted(
        reviews,
        key=lambda item: (item.get("helpful_vote", 0), item.get("rating", 0)),
        reverse=True,
    )
    highlights = []
    for review in sorted_reviews:
        text = review.get("text") or review.get("title")
        if not text:
            continue
        highlights.append(text[:280])
        if len(highlights) >= limit:
            break
    return highlights


def build_product_document(
    product: dict[str, Any], reviews: list[dict[str, Any]], highlights_per_product: int = 5
) -> dict[str, Any]:
    highlights = review_highlights(reviews, highlights_per_product)
    return {
        "product_id": product["product_id"],
        "parent_asin": product["parent_asin"],
        "title": product["title"],
        "category": product["main_category"],
        "brand": product["store"],
        "price": product["price"],
        "average_rating": product["average_rating"],
        "rating_number": product["rating_number"],
        "review_count": len(reviews),
        "features": product.get("features", []),
        "description": product.get("description", ""),
        "image_url": product.get("image_url", ""),
        "review_highlights": highlights,
        "source_text": product_source_text(product, reviews),
    }


def generate_mock_profiles(
    reviews_by_user: dict[str, list[dict[str, Any]]],
    products_by_id: dict[str, dict[str, Any]],
    max_profiles: int = 50,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for user_id, reviews in list(reviews_by_user.items())[:max_profiles]:
        categories = []
        liked_products = []
        recent_text = []
        ratings = []
        for review in reviews:
            product = products_by_id.get(review["parent_asin"])
            if product:
                categories.append(product.get("main_category", ""))
            ratings.append(review.get("rating", 0.0))
            if review.get("rating", 0.0) >= 4.0:
                liked_products.append(review["parent_asin"])
            if review.get("text"):
                recent_text.append(review["text"][:240])
        category_counts = Counter(cat for cat in categories if cat)
        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        profiles[user_id] = {
            "user_id": user_id,
            "preferred_categories": [cat for cat, _ in category_counts.most_common(5)],
            "liked_products": liked_products[:20],
            "rating_count": len(ratings),
            "avg_rating": round(avg_rating, 3),
            "recent_review_text": recent_text[:10],
            "inferred_tags": infer_tags_from_text(recent_text),
        }
    return profiles


def infer_tags_from_text(texts: list[str]) -> list[str]:
    keywords = [
        "durable",
        "battery",
        "sound",
        "quality",
        "easy",
        "comfortable",
        "portable",
        "fast",
        "cheap",
        "expensive",
        "screen",
        "charger",
        "wireless",
    ]
    blob = " ".join(texts).lower()
    return [word for word in keywords if word in blob][:10]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def run_etl(
    meta_path: str | Path,
    reviews_path: str | Path,
    output_dir: str | Path,
    category: str | None = "Electronics",
    max_products: int = 1000,
    max_reviews: int = 10000,
    highlights_per_product: int = 5,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    products: dict[str, dict[str, Any]] = {}
    for raw in iter_jsonl(meta_path):
        product = normalize_metadata_record(raw)
        if not product["parent_asin"] or not product["title"]:
            continue
        if not category_matches(product, category):
            continue
        products[product["parent_asin"]] = product
        if len(products) >= max_products:
            break

    selected_ids = set(products)
    reviews: list[dict[str, Any]] = []
    reviews_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in iter_jsonl(reviews_path):
        review = normalize_review_record(raw)
        if review["parent_asin"] not in selected_ids:
            continue
        reviews.append(review)
        reviews_by_product[review["parent_asin"]].append(review)
        if review["user_id"]:
            reviews_by_user[review["user_id"]].append(review)
        if len(reviews) >= max_reviews:
            break

    documents = [
        build_product_document(product, reviews_by_product.get(parent_asin, []), highlights_per_product)
        for parent_asin, product in products.items()
    ]
    profiles = generate_mock_profiles(reviews_by_user, products)

    product_count = write_jsonl(output / "products.jsonl", products.values())
    review_count = write_jsonl(output / "reviews.jsonl", reviews)
    document_count = write_jsonl(output / "product_documents.jsonl", documents)
    with (output / "mock_user_profiles.json").open("w", encoding="utf-8") as handle:
        json.dump(profiles, handle, ensure_ascii=False, indent=2)

    summary = {
        "category": category,
        "product_count": product_count,
        "review_count": review_count,
        "document_count": document_count,
        "profile_count": len(profiles),
        "output_dir": str(output),
    }
    with (output / "etl_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small Amazon Reviews RAG subset.")
    parser.add_argument("--meta-path", required=True, help="Path to meta_Electronics JSONL or JSONL.GZ")
    parser.add_argument("--reviews-path", required=True, help="Path to Electronics reviews JSONL or JSONL.GZ")
    parser.add_argument("--output-dir", default="../data/processed")
    parser.add_argument("--category", default="Electronics")
    parser.add_argument("--max-products", type=int, default=1000)
    parser.add_argument("--max-reviews", type=int, default=10000)
    args = parser.parse_args()
    summary = run_etl(
        meta_path=args.meta_path,
        reviews_path=args.reviews_path,
        output_dir=args.output_dir,
        category=args.category,
        max_products=args.max_products,
        max_reviews=args.max_reviews,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
