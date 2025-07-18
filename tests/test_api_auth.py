"""Tests for API authentication."""

import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.auth import (
    create_cache_clear_token,
    create_hmac_signature,
    generate_admin_token,
    get_admin_token,
    verify_admin_token,
    verify_cache_clear_permission,
    verify_hmac_signature,
)


class TestAuthUtilities:
    """Test authentication utilities."""

    def test_generate_admin_token(self):
        """Test admin token generation."""
        token1 = generate_admin_token()
        token2 = generate_admin_token()

        # Should be different
        assert token1 != token2

        # Should be URL-safe
        assert all(c.isalnum() or c in "-_" for c in token1)

        # Should be reasonably long
        assert len(token1) >= 32

    def test_get_admin_token_from_env(self):
        """Test getting admin token from environment."""
        test_token = "test-admin-token-123"

        with patch.dict(os.environ, {"BRAIN_GO_BRRR_ADMIN_TOKEN": test_token}):
            token = get_admin_token()
            assert token == test_token

    def test_get_admin_token_generates_if_missing(self):
        """Test admin token generation when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("BRAIN_GO_BRRR_ADMIN_TOKEN", None)

            token = get_admin_token()
            assert token is not None
            assert len(token) >= 32

    def test_hmac_signature_creation(self):
        """Test HMAC signature creation."""
        message = "test message"
        secret = "test secret"

        signature = create_hmac_signature(message, secret)

        # Should be hex string
        assert all(c in "0123456789abcdef" for c in signature)

        # Should be consistent
        signature2 = create_hmac_signature(message, secret)
        assert signature == signature2

        # Different message should give different signature
        signature3 = create_hmac_signature("different message", secret)
        assert signature != signature3

    def test_hmac_signature_verification(self):
        """Test HMAC signature verification."""
        message = "test message"
        secret = "test secret"

        signature = create_hmac_signature(message, secret)

        # Should verify correctly
        assert verify_hmac_signature(message, signature, secret)

        # Should fail with wrong signature
        assert not verify_hmac_signature(message, "wrong-signature", secret)

        # Should fail with wrong secret
        assert not verify_hmac_signature(message, signature, "wrong-secret")

    def test_verify_admin_token_valid(self):
        """Test valid admin token verification."""
        test_token = "test-admin-token-123"

        with patch.dict(os.environ, {"BRAIN_GO_BRRR_ADMIN_TOKEN": test_token}):
            # Should succeed with correct token
            auth_header = f"Bearer {test_token}"
            assert verify_admin_token(auth_header) is True

    def test_verify_admin_token_invalid(self):
        """Test invalid admin token verification."""
        test_token = "test-admin-token-123"

        with patch.dict(os.environ, {"BRAIN_GO_BRRR_ADMIN_TOKEN": test_token}):
            # Should fail with wrong token
            with pytest.raises(HTTPException) as exc_info:
                verify_admin_token("Bearer wrong-token")
            assert exc_info.value.status_code == 403

            # Should fail without Bearer scheme
            with pytest.raises(HTTPException) as exc_info:
                verify_admin_token("wrong-token")
            assert exc_info.value.status_code == 401

            # Should fail without header
            with pytest.raises(HTTPException) as exc_info:
                verify_admin_token(None)
            assert exc_info.value.status_code == 401

    def test_verify_cache_clear_permission_bearer(self):
        """Test cache clear permission with Bearer token."""
        test_token = "test-admin-token-123"

        with patch.dict(os.environ, {"BRAIN_GO_BRRR_ADMIN_TOKEN": test_token}):
            # Should succeed with valid Bearer token
            auth_header = f"Bearer {test_token}"
            assert verify_cache_clear_permission(auth_header) is True

    def test_verify_cache_clear_permission_hmac(self):
        """Test cache clear permission with HMAC signature."""
        # Should accept valid HMAC format
        hmac_sig = "a" * 64  # 64 hex chars
        auth_header = f"HMAC {hmac_sig}"
        assert verify_cache_clear_permission(auth_header) is True

        # Should reject invalid HMAC format
        with pytest.raises(HTTPException) as exc_info:
            verify_cache_clear_permission("HMAC short")
        assert exc_info.value.status_code == 403

    def test_create_cache_clear_token(self):
        """Test cache clear token creation."""
        token = create_cache_clear_token()

        # Should start with HMAC
        assert token.startswith("HMAC ")

        # Should have valid signature format
        signature = token.replace("HMAC ", "")
        assert len(signature) == 64
        assert all(c in "0123456789abcdef" for c in signature)


class TestAPIAuthentication:
    """Test authentication in API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.main import app

        return TestClient(app)

    def test_cache_clear_requires_auth(self, client):
        """Test cache clear endpoint requires authentication."""
        # Should fail without auth
        response = client.delete("/api/v1/cache/clear")
        assert response.status_code == 401

    def test_cache_clear_with_valid_auth(self, client):
        """Test cache clear with valid authentication."""
        test_token = "test-admin-token-123"

        with patch.dict(os.environ, {"BRAIN_GO_BRRR_ADMIN_TOKEN": test_token}):
            response = client.delete(
                "/api/v1/cache/clear", headers={"Authorization": f"Bearer {test_token}"}
            )
            # Should succeed (or return cache unavailable if Redis not running)
            assert response.status_code == 200
            result = response.json()
            assert result["status"] in ["success", "unavailable"]

    def test_cache_clear_with_invalid_auth(self, client):
        """Test cache clear with invalid authentication."""
        response = client.delete(
            "/api/v1/cache/clear", headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 403

    def test_cache_clear_with_hmac_auth(self, client):
        """Test cache clear with HMAC authentication."""
        # Create valid HMAC token
        token = create_cache_clear_token()

        response = client.delete("/api/v1/cache/clear", headers={"Authorization": token})
        # Should succeed (or return cache unavailable if Redis not running)
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "unavailable"]
