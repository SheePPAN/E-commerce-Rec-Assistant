"""
Multi-Agent E-Commerce Recommendation System — FastAPI Entry Point

Endpoints:
  POST /api/v1/recommend          - personalized recommendations
  POST /api/v1/recommend/graph    - recommendations through the LangGraph pipeline
  GET  /api/v1/metrics            - system metrics
  GET  /health                    - health check
"""

from __future__ import annotations

import sys
import os
import asyncio

sys.path.insert(0, os.path.dirname(__file__))

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models.schemas import CatalogReindexRequest, RecommendationRequest, RecommendationResponse
from orchestrator.supervisor import SupervisorOrchestrator
from orchestrator.graph import build_recommendation_graph
from services.metrics import MetricsCollector

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
settings = get_settings()


metrics_collector = MetricsCollector()
supervisor = SupervisorOrchestrator()
rec_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rec_graph
    rec_graph = build_recommendation_graph()
    logger.info("app.startup", model=settings.llm_model)
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent E-Commerce Recommendation System",
    description=(
        "User profile, product recommendation, and recommendation copy agents "
        "for Amazon Electronics RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "healthy", "model": settings.llm_model}


@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Run recommendations with the supervisor orchestrator."""
    response = await supervisor.recommend(request)
    _collect_metrics(response)
    return response


@app.post("/api/v1/recommend/graph")
async def recommend_via_graph(request: RecommendationRequest):
    """Run recommendations through the LangGraph state graph."""
    if not rec_graph:
        return {"error": "Graph not initialized"}
    state = {
        "user_id": request.user_id,
        "query": request.query,
        "scene": request.scene,
        "num_items": request.num_items,
        "context": request.context,
    }
    result = await rec_graph.ainvoke(state)
    return {
        "request_id": result.get("request_id"),
        "user_id": result.get("user_id"),
        "products": [p.model_dump() for p in result.get("final_products", [])],
        "marketing_copies": result.get("marketing_copies", []),
        "recommendation_summary": result.get("recommendation_summary", ""),
        "total_latency_ms": round(result.get("total_latency_ms", 0), 1),
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Return system metrics."""
    return {
        "agents": metrics_collector.get_agent_stats(),
        "business": metrics_collector.get_business_stats(),
    }


@app.get("/api/v1/catalog/status")
async def catalog_status():
    """Report whether the Milvus-backed Amazon catalog index is available."""
    return supervisor.product_rec_agent.catalog.status()


@app.post("/api/v1/catalog/reindex")
async def catalog_reindex(request: CatalogReindexRequest):
    """Local/dev utility: index processed Amazon product documents into Milvus."""
    result = await asyncio.to_thread(
        supervisor.product_rec_agent.catalog.reindex_from_processed,
        request.processed_dir,
        request.limit,
        request.recreate,
    )
    return result


def _collect_metrics(response: RecommendationResponse):
    for name, result in response.agent_results.items():
        metrics_collector.record_agent_call(
            agent_name=name,
            success=result.success,
            latency_ms=result.latency_ms,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
