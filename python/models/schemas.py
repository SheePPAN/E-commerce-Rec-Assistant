from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class UserSegment(str, Enum):
    NEW_USER = "new_user"
    ACTIVE = "active"
    HIGH_VALUE = "high_value"
    PRICE_SENSITIVE = "price_sensitive"
    CHURN_RISK = "churn_risk"


class UserProfile(BaseModel):
    user_id: str
    age: int | None = None
    gender: str | None = None
    city: str | None = None
    segments: list[UserSegment] = Field(default_factory=list)
    preferred_categories: list[str] = Field(default_factory=list)
    price_range: tuple[float, float] = (0.0, 10000.0)
    recent_views: list[str] = Field(default_factory=list)
    recent_purchases: list[str] = Field(default_factory=list)
    rfm_score: dict[str, float] = Field(default_factory=dict)
    real_time_tags: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    description: str = ""
    brand: str = ""
    seller_id: str = ""
    tags: list[str] = Field(default_factory=list)
    score: float = 0.0
    image_url: str = ""
    average_rating: float | None = None
    rating_number: int = 0
    review_count: int = 0
    review_highlights: list[str] = Field(default_factory=list)
    semantic_score: float = 0.0
    source_parent_asin: str = ""


class RecommendationRequest(BaseModel):
    user_id: str
    query: str | None = None
    scene: str = "homepage"
    num_items: int = 10
    context: dict[str, Any] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent_name: str
    success: bool = True
    latency_ms: float = 0.0
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class UserProfileResult(AgentResult):
    agent_name: str = "user_profile"
    profile: UserProfile | None = None


class ProductRecResult(AgentResult):
    agent_name: str = "product_rec"
    products: list[Product] = Field(default_factory=list)
    recall_strategy: str = ""


class MarketingCopyResult(AgentResult):
    agent_name: str = "marketing_copy"
    copies: list[dict[str, str]] = Field(default_factory=list)
    prompt_template_used: str = ""


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    products: list[Product] = Field(default_factory=list)
    marketing_copies: list[dict[str, str]] = Field(default_factory=list)
    recommendation_summary: str = ""
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    total_latency_ms: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)


class CatalogReindexRequest(BaseModel):
    processed_dir: str = "data/processed"
    limit: int | None = None
    recreate: bool = False


class CatalogStatus(BaseModel):
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    available: bool = False
    loaded: bool = False
    entity_count: int = 0
    error: str | None = None
