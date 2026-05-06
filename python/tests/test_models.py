from models.schemas import Product, RecommendationRequest, RecommendationResponse


def test_recommendation_request_accepts_old_shape_without_query():
    request = RecommendationRequest(user_id="user_001", num_items=3)
    assert request.query is None
    assert request.scene == "homepage"


def test_product_serializes_amazon_review_fields():
    product = Product(
        product_id="B001",
        source_parent_asin="B001",
        name="Wireless Headphones",
        category="Electronics",
        price=49.99,
        average_rating=4.4,
        rating_number=1200,
        review_count=25,
        review_highlights=["Comfortable for travel"],
        semantic_score=0.88,
    )
    payload = product.model_dump()
    assert payload["average_rating"] == 4.4
    assert payload["review_highlights"] == ["Comfortable for travel"]
    assert payload["semantic_score"] == 0.88


def test_recommendation_response_has_summary_default():
    response = RecommendationResponse(request_id="r1", user_id="u1")
    assert response.recommendation_summary == ""
