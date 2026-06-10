import secrets
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    
    # We use compare_digest to prevent timing attacks
    if not secrets.compare_digest(api_key_header, settings.INTERNAL_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return api_key_header

import time
from collections import defaultdict
from fastapi import Request

# In-memory rate limiter (10 requests / min)
# Note: For production with multiple workers, Redis is recommended.
_rate_limit_records = defaultdict(list)

def rate_limit_dependency(request: Request):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean up old entries (older than 60 seconds)
    _rate_limit_records[client_ip] = [t for t in _rate_limit_records[client_ip] if now - t < 60]
    
    if len(_rate_limit_records[client_ip]) >= 10:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 10 requests per minute."
        )
    
    _rate_limit_records[client_ip].append(now)
    return True

