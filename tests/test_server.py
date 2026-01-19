"""Tests for the MCP server endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.server import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test /health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    def test_root_endpoint(self, client):
        """Test / endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RLM MCP Server"
        assert "version" in data
        assert data["docs"] == "/docs"


class TestMCPEndpoints:
    """Tests for MCP tool endpoints."""

    def test_mcp_requires_api_key(self, client):
        """Test that MCP endpoint requires API key."""
        response = client.post(
            "/v1/test-project/mcp",
            json={"tool": "rlm_stats", "params": {}},
        )
        # Should fail with 422 (missing header) or 401 (invalid key)
        assert response.status_code in [401, 422]

    def test_mcp_invalid_api_key(self, client):
        """Test MCP endpoint with invalid API key."""
        response = client.post(
            "/v1/test-project/mcp",
            json={"tool": "rlm_stats", "params": {}},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
        assert "Invalid API key" in data["error"]

    def test_context_requires_api_key(self, client):
        """Test that context endpoint requires API key."""
        response = client.get("/v1/test-project/context")
        assert response.status_code in [401, 422]

    def test_limits_requires_api_key(self, client):
        """Test that limits endpoint requires API key."""
        response = client.get("/v1/test-project/limits")
        assert response.status_code in [401, 422]


class TestRequestValidation:
    """Tests for request validation."""

    def test_invalid_tool_name(self, client):
        """Test that invalid tool names are rejected."""
        response = client.post(
            "/v1/test-project/mcp",
            json={"tool": "invalid_tool", "params": {}},
            headers={"X-API-Key": "test-key"},
        )
        # Should fail validation (422) or auth (401)
        assert response.status_code in [401, 422]

    def test_missing_tool(self, client):
        """Test that missing tool field is rejected."""
        response = client.post(
            "/v1/test-project/mcp",
            json={"params": {}},
            headers={"X-API-Key": "test-key"},
        )
        assert response.status_code == 422


class TestResponseFormat:
    """Tests for response format consistency."""

    def test_error_response_format(self, client):
        """Test that error responses have consistent format."""
        response = client.post(
            "/v1/test-project/mcp",
            json={"tool": "rlm_stats", "params": {}},
            headers={"X-API-Key": "invalid-key"},
        )
        data = response.json()

        # Check required fields in error response
        assert "success" in data
        assert data["success"] is False
        assert "error" in data
        assert "usage" in data
        assert "latency_ms" in data["usage"]
