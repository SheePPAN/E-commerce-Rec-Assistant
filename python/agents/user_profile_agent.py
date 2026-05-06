"""
User profile agent for the Amazon Electronics shopping assistant.
- Real-time feature extraction from views, clicks, purchases, and saved items.
- User segmentation with RFM-style signals and live tags.
- Profile merging from online context and optional feature-store data.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    AgentResult,
    UserProfile,
    UserProfileResult,
    UserSegment,
)

from .base_agent import BaseAgent

SYSTEM_PROMPT = """You are a user profiling expert for an Amazon Electronics shopping assistant. Analyze the user's behavior data and infer a lightweight shopping profile.

Return the following JSON shape:
{
  "segments": ["new_user"|"active"|"high_value"|"price_sensitive"|"churn_risk"],
  "preferred_categories": ["category 1", "category 2"],
  "price_range": [minimum_usd, maximum_usd],
  "rfm_score": {"recency": 0-1, "frequency": 0-1, "monetary": 0-1},
  "real_time_tags": {"active_period": "...", "shopping_intent": "...", "device_affinity": "..."}
}

Use English category and tag names, and interpret prices as USD. Only output JSON. Do not include any other text."""


class UserProfileAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="user_profile",
            timeout=settings.agent_timeout_user_profile,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.6,
            max_tokens=1024,
        )
        self.feature_store: Any = None  # injected in Phase 2

    async def _execute(self, **kwargs: Any) -> UserProfileResult:
        user_id: str = kwargs["user_id"]
        context: dict = kwargs.get("context", {})

        behavior_data = await self._collect_behavior(user_id, context)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User ID: {user_id}\nBehavior data: {json.dumps(behavior_data, ensure_ascii=False)}"),
        ]
        response = await self.llm.ainvoke(messages)

        profile_data = self._parse_profile(user_id, response.content)

        return UserProfileResult(
            success=True,
            profile=profile_data,
            data={"raw_analysis": response.content},
            confidence=0.85,
        )

    async def _collect_behavior(self, user_id: str, context: dict) -> dict:
        """Collect user behavior from feature store or context fallback."""
        if self.feature_store:
            return await self.feature_store.get_user_features(user_id)
        return {
            "user_id": user_id,
            "recent_views": context.get(
                "recent_views",
                ["wireless headphones", "USB-C chargers", "tablet accessories"],
            ),
            "recent_purchases": context.get("recent_purchases", ["charging cable"]),
            "view_count_7d": context.get("view_count_7d", 25),
            "purchase_count_30d": context.get("purchase_count_30d", 3),
            "avg_order_amount": context.get("avg_order_amount", 79.0),
            "active_hours": context.get("active_hours", [20, 21, 22]),
        }

    def _parse_profile(self, user_id: str, raw: str) -> UserProfile:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            data = {}

        segments = []
        for s in data.get("segments", ["active"]):
            try:
                segments.append(UserSegment(s))
            except ValueError:
                continue

        price_range_raw = data.get("price_range", [0, 1000])
        price_range = (
            float(price_range_raw[0]),
            float(price_range_raw[1]) if len(price_range_raw) > 1 else 1000.0,
        )

        return UserProfile(
            user_id=user_id,
            segments=segments or [UserSegment.ACTIVE],
            preferred_categories=data.get("preferred_categories", []),
            price_range=price_range,
            rfm_score=data.get("rfm_score", {}),
            real_time_tags=data.get("real_time_tags", {}),
        )
