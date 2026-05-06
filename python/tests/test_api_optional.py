import pytest


pytest.importorskip("fastapi")
pytest.importorskip("langchain_openai")

from fastapi.testclient import TestClient
from main import app


def test_catalog_status_endpoint_shape():
    client = TestClient(app)
    response = client.get("/api/v1/catalog/status")
    assert response.status_code == 200
    payload = response.json()
    assert "collection_name" in payload
    assert "available" in payload
