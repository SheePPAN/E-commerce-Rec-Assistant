"""
Supervisor orchestrator: parallel dispatch plus aggregation.

                    ┌──────────────┐
                    │  Supervisor   │
                    └──────┬───────┘
           ┌───────┬───────┬────────┐
           ▼       ▼       ▼        │
      UserProfile  ProdRec  MktCopy │
           │       │       │        │
           └───────┴───────┘        │
                    │               │
                    ▼               │
               Aggregator ◄─────────┘
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from agents import (
    MarketingCopyAgent,
    ProductRecAgent,
    UserProfileAgent,
)
from models.schemas import (
    Product,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
)

try:
    import structlog

    logger = structlog.get_logger()
except ModuleNotFoundError:
    import logging

    class _KeywordLogger:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def info(self, event: str, **kwargs):
            self.wrapped.info("%s %s", event, kwargs)

        def error(self, event: str, **kwargs):
            self.wrapped.error("%s %s", event, kwargs)

    logger = _KeywordLogger(logging.getLogger(__name__))


class SupervisorOrchestrator:
    """Coordinates three agents in parallel-then-aggregate pattern."""

    def __init__(self):
        self.user_profile_agent = UserProfileAgent()
        self.product_rec_agent = ProductRecAgent()
        self.marketing_copy_agent = MarketingCopyAgent()

    async def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()

        logger.info(
            "supervisor.start",
            request_id=request_id,
            user_id=request.user_id,
            scene=request.scene,
        )

        # Phase 1: parallel — user profile + product recall
        profile_result, rec_result = await asyncio.gather(
            self.user_profile_agent.run(
                user_id=request.user_id,
                context=request.context,
            ),
            self.product_rec_agent.run(
                user_profile=None,
                num_items=request.num_items * 2,
                query=request.query,
            ),
        )

        user_profile: UserProfile | None = getattr(profile_result, "profile", None)
        raw_products: list[Product] = getattr(rec_result, "products", [])

        # Phase 2: re-rank with profile. Items are unlimited, so TopN is used directly.
        rerank_result = await self.product_rec_agent.run(
            user_profile=user_profile,
            num_items=request.num_items,
            query=request.query,
            candidates=raw_products,
            recall_strategy=getattr(rec_result, "recall_strategy", ""),
        )

        ranked_products: list[Product] = getattr(rerank_result, "products", raw_products)
        final_products = ranked_products[:request.num_items]

        # Phase 3: marketing copy generation with final product list
        copy_result = await self.marketing_copy_agent.run(
            user_profile=user_profile,
            products=final_products,
        )
        copies = getattr(copy_result, "copies", [])
        recommendation_summary = await self.marketing_copy_agent.summarize_recommendations(
            query=request.query,
            user_profile=user_profile,
            products=final_products,
        )

        total_latency = (time.perf_counter() - start) * 1000

        logger.info(
            "supervisor.complete",
            request_id=request_id,
            total_latency_ms=round(total_latency, 1),
            product_count=len(final_products),
            copy_count=len(copies),
        )

        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            products=final_products,
            marketing_copies=copies,
            recommendation_summary=recommendation_summary,
            agent_results={
                "user_profile": profile_result,
                "product_rec": rerank_result,
                "marketing_copy": copy_result,
            },
            total_latency_ms=total_latency,
        )
