"""
Recommendation copy agent for Amazon Electronics products.
- Selects an English prompt template from the inferred user segment.
- Generates concise, review-grounded shopping guidance for each product.
- Keeps copy objective instead of promotional.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import get_settings
from models.schemas import (
    MarketingCopyResult,
    Product,
    UserProfile,
    UserSegment,
)

from .base_agent import BaseAgent

PROMPT_TEMPLATES = {
    UserSegment.NEW_USER: """You are a shopping decision assistant. For each Amazon Electronics product, write one objective recommendation based on the product metadata, ratings, and review highlights. Explain who it fits, the main benefit, and one limitation to consider. Avoid promotional hype. Keep each item under 55 words.""",

    UserSegment.HIGH_VALUE: """You are a shopping decision assistant. For each Amazon Electronics product, write one objective recommendation based on metadata, ratings, and review highlights. Emphasize quality, reliability, and whether it is worth considering for a higher-budget shopper. Avoid promotional hype. Keep each item under 55 words.""",

    UserSegment.PRICE_SENSITIVE: """You are a shopping decision assistant. For each Amazon Electronics product, write one objective recommendation based on metadata, ratings, and review highlights. Focus on value, repeated positive review themes, and tradeoffs. Avoid promotional hype. Keep each item under 55 words.""",

    UserSegment.ACTIVE: """You are a shopping decision assistant. For each Amazon Electronics product, write one objective recommendation based on metadata, ratings, and review highlights. Explain the best use case, repeated review strengths, and what to check before buying. Avoid promotional hype. Keep each item under 55 words.""",

    UserSegment.CHURN_RISK: """You are a shopping decision assistant. For each Amazon Electronics product, write one objective recommendation based on metadata, ratings, and review highlights. Help the shopper quickly decide whether the product is worth renewed attention. Avoid promotional hype. Keep each item under 55 words.""",
}

COPY_OUTPUT_INSTRUCTION = """
Return a JSON array with this shape:
[{"product_id": "xxx", "copy": "recommendation text"}]
Only output JSON. Do not include any other text."""


class MarketingCopyAgent(BaseAgent):
    def __init__(self):
        settings = get_settings()
        super().__init__(
            name="marketing_copy",
            timeout=settings.agent_timeout_marketing_copy,
        )
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=0.9,
            max_tokens=2048,
        )

    async def _execute(self, **kwargs: Any) -> MarketingCopyResult:
        user_profile: UserProfile | None = kwargs.get("user_profile")
        products: list[Product] = kwargs.get("products", [])

        if not products:
            return MarketingCopyResult(success=True, copies=[], confidence=1.0)

        template_key = self._select_template(user_profile)
        system_prompt = PROMPT_TEMPLATES[template_key]

        product_info = "\n".join(
            (
                f"- ID:{p.product_id} name:{p.name} category:{p.category} "
                f"price:${p.price:.2f} tags:{','.join(p.tags)} "
                f"rating:{p.average_rating} review_count:{p.review_count} "
                f"review_highlights:{json.dumps(p.review_highlights[:3], ensure_ascii=False)}"
            )
            for p in products
        )

        messages = [
            SystemMessage(content=system_prompt + COPY_OUTPUT_INSTRUCTION),
            HumanMessage(content=f"Product list:\n{product_info}"),
        ]
        response = await self.llm.ainvoke(messages)

        copies = self._parse_copies(response.content)

        return MarketingCopyResult(
            success=True,
            copies=copies,
            prompt_template_used=template_key.value,
            data={"raw_response": response.content},
            confidence=0.9,
        )

    async def summarize_recommendations(
        self,
        query: str | None,
        user_profile: UserProfile | None,
        products: list[Product],
    ) -> str:
        if not products:
            return "No matching products were found for this request."

        product_payload = [
            {
                "id": product.product_id,
                "name": product.name,
                "brand": product.brand,
                "category": product.category,
                "price": product.price,
                "average_rating": product.average_rating,
                "rating_number": product.rating_number,
                "review_count": product.review_count,
                "review_highlights": product.review_highlights[:3],
            }
            for product in products
        ]
        profile_payload = {
            "segments": [segment.value for segment in user_profile.segments] if user_profile else [],
            "preferred_categories": user_profile.preferred_categories if user_profile else [],
            "price_range": list(user_profile.price_range) if user_profile else [],
        }
        messages = [
            SystemMessage(
                content=(
                    "You are a shopping assistant. Write a concise recommendation summary "
                    "grounded only in the provided product metadata and review highlights. "
                    "Mention why the top products match the query. Keep it under 120 words."
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "query": query or "",
                        "user_profile": profile_payload,
                        "products": product_payload,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        try:
            response = await self.llm.ainvoke(messages)
            summary = str(response.content).strip()
            if summary:
                return summary
        except Exception:
            pass
        names = ", ".join(product.name for product in products[:3])
        highlight = ""
        for product in products:
            if product.review_highlights:
                highlight = product.review_highlights[0]
                break
        if query:
            return f"For '{query}', the strongest matches are {names}. Review evidence: {highlight}"
        return f"Recommended products based on the available profile and review signals: {names}."

    def _select_template(self, profile: UserProfile | None) -> UserSegment:
        if not profile or not profile.segments:
            return UserSegment.ACTIVE
        priority = [
            UserSegment.NEW_USER,
            UserSegment.HIGH_VALUE,
            UserSegment.CHURN_RISK,
            UserSegment.PRICE_SENSITIVE,
            UserSegment.ACTIVE,
        ]
        for seg in priority:
            if seg in profile.segments:
                return seg
        return UserSegment.ACTIVE

    def _parse_copies(self, raw: str) -> list[dict[str, str]]:
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            return []
