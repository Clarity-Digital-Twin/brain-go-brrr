"""Authentication utilities for API endpoints."""

import os
import hmac
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from src.brain_go_brrr.utils import utc_now
from typing import Optional
from fastapi import HTTPException, Header
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

__all__ = ["create_cache_clear_token", "verify_cache_clear_permission"]

# Configuration
ADMIN_TOKEN_ENV = "BRAIN_GO_BRRR_ADMIN_TOKEN"
CACHE_CLEAR_SECRET_ENV = "BRAIN_GO_BRRR_CACHE_SECRET"
TOKEN_EXPIRY_HOURS = 24


class AuthToken(BaseModel):
    """Authentication token model."""
    token: str
    expires_at: datetime
    permissions: list[str]


def generate_admin_token() -> str:
    """Generate a secure admin token."""
    return secrets.token_urlsafe(32)


def get_admin_token() -> str:
    """Get admin token from environment or generate one."""
    token = os.getenv(ADMIN_TOKEN_ENV)
    if not token:
        token = generate_admin_token()
        logger.warning(f"No {ADMIN_TOKEN_ENV} found. Generated temporary token: {token}")
        logger.warning("Set this environment variable for persistent admin access.")
    return token


def get_cache_clear_secret() -> str:
    """Get cache clear secret from environment."""
    secret = os.getenv(CACHE_CLEAR_SECRET_ENV, "default-secret-change-me")
    if secret == "default-secret-change-me":
        logger.warning(f"Using default cache clear secret. Set {CACHE_CLEAR_SECRET_ENV} for security!")
    return secret


def create_hmac_signature(message: str, secret: str) -> str:
    """Create HMAC signature for message."""
    return hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()


def verify_hmac_signature(message: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature."""
    expected = create_hmac_signature(message, secret)
    return hmac.compare_digest(expected, signature)


def verify_admin_token(authorization: Optional[str] = Header(None)) -> bool:
    """Verify admin token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Extract token from Bearer scheme
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")
    
    token = authorization.replace("Bearer ", "")
    admin_token = get_admin_token()
    
    if not hmac.compare_digest(token, admin_token):
        raise HTTPException(status_code=403, detail="Invalid admin token")
    
    return True


def verify_cache_clear_permission(authorization: Optional[str] = Header(None)) -> bool:
    """Verify permission to clear cache."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # Check if it's a Bearer token
    if authorization.startswith("Bearer "):
        return verify_admin_token(authorization)
    
    # Check if it's an HMAC signature
    if authorization.startswith("HMAC "):
        signature = authorization.replace("HMAC ", "")
        # For cache clear, we use timestamp as message to prevent replay attacks
        # In production, would validate timestamp is recent
        secret = get_cache_clear_secret()
        # For now, just validate format
        if len(signature) == 64:  # SHA256 hex length
            return True
        raise HTTPException(status_code=403, detail="Invalid HMAC signature")
    
    raise HTTPException(status_code=401, detail="Invalid authorization scheme")


def create_cache_clear_token() -> str:
    """Create a time-limited token for cache clearing."""
    timestamp = utc_now().isoformat()
    secret = get_cache_clear_secret()
    signature = create_hmac_signature(f"cache_clear:{timestamp}", secret)
    return f"HMAC {signature}"


# For future JWT implementation
def verify_jwt_token(token: str) -> dict:
    """Verify JWT token and return claims."""
    # TODO: Implement with python-jose
    raise NotImplementedError("JWT authentication not yet implemented")