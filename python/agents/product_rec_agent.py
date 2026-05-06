"""
Amazon Electronics recommendation agent.
- Recall: Milvus semantic search over processed Amazon review documents.
- Ranking: LLM reranking using profile signals, product metadata, and review evidence.
- Fallback: English Amazon Electronics seed catalog when no query/index result is available.
"""

from __future__ import annotations

import json
import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import Product, ProductRecResult, UserProfile
from services.catalog_index import AmazonCatalogIndex

from .base_agent import BaseAgent

SEMANTIC_RECALL_STRATEGY = "amazon_reviews_milvus_semantic"
FALLBACK_RECALL_STRATEGY = "amazon_electronics_profile_fallback"
PROVIDED_CANDIDATES_STRATEGY = "provided_candidates"

RERANK_PROMPT = """You are an Amazon Electronics recommendation ranking expert. Re-rank the candidate products and select the best {num_items} items using the user query, user profile, product metadata, and review highlights.

User query:
{query}

User profile:
{user_profile}

Candidate products:
{candidates}

Ranking guidelines:
1. Prioritize semantic fit with the user query.
2. Prefer products whose review highlights support the recommendation.
3. Use average rating, rating count, review count, and semantic score as quality signals.
4. Use preferred categories and price range as personalization signals.
5. Avoid over-repeating the same category or brand when similarly strong options exist.

Return the product IDs as a JSON array, ordered from strongest to weakest recommendation:
["product_id_1", "product_id_2", ...]

Only output the JSON array. Do not include any other text."""

MOCK_PRODUCTS = [
    Product(
        product_id="AMZ-FALLBACK-001",
        source_parent_asin="AMZ-FALLBACK-001",
        name="Noise Cancelling Bluetooth Headphones",
        category="All Electronics",
        price=89.99,
        brand="SoundCore",
        seller_id="amazon-electronics-fallback",
        tags=["noise cancellation", "bluetooth", "travel"],
        average_rating=4.4,
        rating_number=8200,
        review_count=125,
        review_highlights=["Reviewers often mention long battery life and a comfortable fit."],
    ),
    Product(
        product_id="AMZ-FALLBACK-002",
        source_parent_asin="AMZ-FALLBACK-002",
        name="65W USB-C GaN Fast Charger",
        category="Cell Phones & Accessories",
        price=29.99,
        brand="Anker",
        seller_id="amazon-electronics-fallback",
        tags=["usb-c", "fast charging", "portable"],
        average_rating=4.6,
        rating_number=15400,
        review_count=210,
        review_highlights=["Reviews praise the compact size and reliable laptop charging."],
    ),
    Product(
        product_id="AMZ-FALLBACK-003",
        source_parent_asin="AMZ-FALLBACK-003",
        name="Portable Bluetooth Speaker",
        category="Portable Audio & Video",
        price=49.99,
        brand="JBL",
        seller_id="amazon-electronics-fallback",
        tags=["speaker", "water resistant", "outdoor"],
        average_rating=4.5,
        rating_number=23400,
        review_count=180,
        review_highlights=["Buyers like the loud sound for its size and easy pairing."],
    ),
    Product(
        product_id="AMZ-FALLBACK-004",
        source_parent_asin="AMZ-FALLBACK-004",
        name="27-Inch QHD USB-C Monitor",
        category="Computers",
        price=249.99,
        brand="Dell",
        seller_id="amazon-electronics-fallback",
        tags=["monitor", "usb-c", "home office"],
        average_rating=4.3,
        rating_number=3700,
        review_count=95,
        review_highlights=["Reviewers call out sharp text and convenient single-cable docking."],
    ),
    Product(
        product_id="AMZ-FALLBACK-005",
        source_parent_asin="AMZ-FALLBACK-005",
        name="WiFi 6 Mesh Router Kit",
        category="Computers",
        price=179.99,
        brand="TP-Link",
        seller_id="amazon-electronics-fallback",
        tags=["wifi 6", "mesh", "home network"],
        average_rating=4.4,
        rating_number=9100,
        review_count=140,
        review_highlights=["Many reviews mention broader coverage and straightforward setup."],
    ),
    Product(
        product_id="AMZ-FALLBACK-006",
        source_parent_asin="AMZ-FALLBACK-006",
        name="4K HDMI Switch",
        category="Home Audio & Theater",
        price=34.99,
        brand="Kinivo",
        seller_id="amazon-electronics-fallback",
        tags=["hdmi", "4k", "tv accessories"],
        average_rating=4.2,
        rating_number=5100,
        review_count=75,
        review_highlights=["Reviews say it is useful for connecting multiple consoles or streamers."],
    ),
    Product(
        product_id="AMZ-FALLBACK-007",
        source_parent_asin="AMZ-FALLBACK-007",
        name="Smartwatch Silicone Replacement Band",
        category="Wearable Technology",
        price=12.99,
        brand="NotoCity",
        seller_id="amazon-electronics-fallback",
        tags=["smartwatch", "replacement band", "fitness"],
        average_rating=4.3,
        rating_number=2300,
        review_count=65,
        review_highlights=["Customers like the quick-release pins and comfortable silicone."],
    ),
    Product(
        product_id="AMZ-FALLBACK-008",
        source_parent_asin="AMZ-FALLBACK-008",
        name="Laptop Protective Skin",
        category="Computers",
        price=19.99,
        brand="Digi-Tatoo",
        seller_id="amazon-electronics-fallback",
        tags=["laptop", "skin", "scratch protection"],
        average_rating=4.1,
        rating_number=1200,
        review_count=45,
        review_highlights=["Reviewers appreciate the look, but some warn to check model compatibility."],
    ),
    Product(
        product_id="AMZ-FALLBACK-009",
        source_parent_asin="AMZ-FALLBACK-009",
        name="Action Camera Accessory Kit",
        category="Camera & Photo",
        price=39.99,
        brand="GoPro Compatible",
        seller_id="amazon-electronics-fallback",
        tags=["camera", "mounts", "travel"],
        average_rating=4.2,
        rating_number=6400,
        review_count=88,
        review_highlights=["Buyers like the range of mounts for travel and outdoor recording."],
    ),
    Product(
        product_id="AMZ-FALLBACK-010",
        source_parent_asin="AMZ-FALLBACK-010",
        name="Bluetooth Ergonomic Mouse",
        category="Computer Accessories & Peripherals",
        price=39.99,
        brand="Logitech",
        seller_id="amazon-electronics-fallback",
        tags=["mouse", "bluetooth", "productivity"],
        average_rating=4.5,
        rating_number=18700,
        review_count=160,
        review_highlights=["Reviews highlight quiet clicks and comfortable all-day use."],
    ),
    Product(
        product_id="AMZ-FALLBACK-011",
        source_parent_asin="AMZ-FALLBACK-011",
        name="2TB Portable SSD",
        category="Computers",
        price=129.99,
        brand="Samsung",
        seller_id="amazon-electronics-fallback",
        tags=["ssd", "storage", "portable"],
        average_rating=4.7,
        rating_number=31200,
        review_count=230,
        review_highlights=["Customers praise fast transfers and the compact enclosure."],
    ),
    Product(
        product_id="AMZ-FALLBACK-012",
        source_parent_asin="AMZ-FALLBACK-012",
        name="USB-C Docking Station",
        category="Computer Accessories & Peripherals",
        price=79.99,
        brand="Satechi",
        seller_id="amazon-electronics-fallback",
        tags=["usb-c", "dock", "home office"],
        average_rating=4.3,
        rating_number=7600,
        review_count=115,
        review_highlights=["Reviews mention useful ports, though power delivery depends on the laptop."],
    ),
    Product(
        product_id="AMZ-FALLBACK-013",
        source_parent_asin="AMZ-FALLBACK-013",
        name="Dash Cam with Night Vision",
        category="Car Electronics",
        price=69.99,
        brand="Vantrue",
        seller_id="amazon-electronics-fallback",
        tags=["dash cam", "night vision", "driving"],
        average_rating=4.2,
        rating_number=9800,
        review_count=102,
        review_highlights=["Drivers like the clear daytime footage and compact windshield mount."],
    ),
    Product(
        product_id="AMZ-FALLBACK-014",
        source_parent_asin="AMZ-FALLBACK-014",
        name="Streaming Media Player",
        category="Television & Video",
        price=39.99,
        brand="Roku",
        seller_id="amazon-electronics-fallback",
        tags=["streaming", "4k", "tv"],
        average_rating=4.6,
        rating_number=42000,
        review_count=260,
        review_highlights=["Reviewers like the simple interface and broad app support."],
    ),
    Product(
        product_id="AMZ-FALLBACK-015",
        source_parent_asin="AMZ-FALLBACK-015",
        name="Rechargeable AA Battery Charger",
        category="Accessories & Supplies",
        price=24.99,
        brand="Panasonic",
        seller_id="amazon-electronics-fallback",
        tags=["batteries", "charger", "household electronics"],
        average_rating=4.5,
        rating_number=19800,
        review_count=145,
        review_highlights=["Customers value the reusable batteries for remotes, toys, and accessories."],
    ),
]


class ProductRecAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="product_rec",
            timeout=settings.agent_timeout_product_rec,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.3,
            max_tokens=512,
        )
        self.catalog = AmazonCatalogIndex()
        self.vector_store: Any = self.catalog

    async def _execute(self, **kwargs: Any) -> ProductRecResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        num_items: int = kwargs.get("num_items", 10)
        query: str = kwargs.get("query") or ""
        provided_candidates: list[Product] | None = kwargs.get("candidates")
        provided_strategy: str = kwargs.get("recall_strategy") or ""

        if provided_candidates is not None:
            candidates = provided_candidates
            recall_strategy = provided_strategy or PROVIDED_CANDIDATES_STRATEGY
        else:
            candidates, recall_strategy = await self._recall(
                user_profile,
                num_items * 3,
                query=query,
            )
        ranked_ids = await self._rerank(user_profile, candidates, num_items, query=query)

        id_to_product = {p.product_id: p for p in candidates}
        final_products = []
        for pid in ranked_ids:
            if pid in id_to_product:
                final_products.append(id_to_product[pid])
        if len(final_products) < num_items:
            for p in candidates:
                if p.product_id not in ranked_ids:
                    final_products.append(p)
                    if len(final_products) >= num_items:
                        break

        return ProductRecResult(
            success=True,
            products=final_products[:num_items],
            recall_strategy=recall_strategy,
            data={"candidate_count": len(candidates), "reranked": len(ranked_ids)},
            confidence=0.8,
        )

    async def _recall(
        self, profile: UserProfile | None, limit: int, query: str = ""
    ) -> tuple[list[Product], str]:
        """Recall products with semantic search first, then an English Electronics fallback catalog."""
        if query.strip():
            products = self.catalog.search(query, limit=limit)
            if products:
                return products, SEMANTIC_RECALL_STRATEGY

        candidates = list(MOCK_PRODUCTS)
        if profile and profile.preferred_categories:
            preferred = set(profile.preferred_categories)
            candidates.sort(
                key=lambda p: (p.category in preferred, random.random()),
                reverse=True,
            )

        return candidates[:limit], FALLBACK_RECALL_STRATEGY

    async def _rerank(
        self,
        profile: UserProfile | None,
        candidates: list[Product],
        num_items: int,
        query: str = "",
    ) -> list[str]:
        if not profile and not query:
            return [p.product_id for p in candidates[:num_items]]

        profile_summary = {
            "segments": [s.value for s in profile.segments] if profile else [],
            "preferred_categories": profile.preferred_categories if profile else [],
            "price_range": list(profile.price_range) if profile else [0, 1000],
        }
        candidate_summary = [
            {
                "id": p.product_id,
                "name": p.name,
                "category": p.category,
                "price": p.price,
                "brand": p.brand,
                "tags": p.tags,
                "average_rating": p.average_rating,
                "rating_number": p.rating_number,
                "review_count": p.review_count,
                "semantic_score": p.semantic_score,
                "review_highlights": p.review_highlights[:3],
            }
            for p in candidates
        ]
        prompt = RERANK_PROMPT.format(
            num_items=num_items,
            query=query
            or "No natural-language query was provided. Rank by user profile and review-backed quality signals.",
            user_profile=json.dumps(profile_summary, ensure_ascii=False),
            candidates=json.dumps(candidate_summary, ensure_ascii=False),
        )
        messages = [
            SystemMessage(content="You are an Amazon Electronics recommendation ranking expert."),
            HumanMessage(content=prompt),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except Exception:
            return [p.product_id for p in candidates[:num_items]]
