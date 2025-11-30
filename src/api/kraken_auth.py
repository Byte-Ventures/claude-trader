"""
Kraken API authentication using HMAC-SHA512.

Kraken uses a nonce-based HMAC-SHA512 signature scheme:
1. Create a nonce (increasing integer, usually timestamp in ms)
2. SHA256 hash of: nonce + encoded POST data
3. HMAC-SHA512 of: path + SHA256 hash, using base64-decoded secret
4. Base64 encode the result for the API-Sign header
"""

import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Optional


def get_nonce() -> str:
    """
    Generate a unique nonce for Kraken API requests.

    Kraken requires an increasing nonce for each request.
    Using microsecond timestamp ensures uniqueness.

    Returns:
        Nonce as string (microsecond timestamp)
    """
    return str(int(time.time() * 1000000))


def create_signature(
    api_secret: str,
    url_path: str,
    nonce: str,
    post_data: Optional[Dict] = None,
) -> str:
    """
    Create HMAC-SHA512 signature for Kraken API request.

    Args:
        api_secret: Base64-encoded API secret from Kraken
        url_path: API endpoint path (e.g., /0/private/Balance)
        nonce: Unique nonce for this request
        post_data: POST data dictionary (will be URL-encoded)

    Returns:
        Base64-encoded signature for API-Sign header
    """
    # Prepare POST data with nonce
    if post_data is None:
        post_data = {}

    post_data["nonce"] = nonce

    # URL-encode the POST data
    encoded_post = urllib.parse.urlencode(post_data)

    # SHA256 hash of nonce + encoded POST data
    message = (nonce + encoded_post).encode("utf-8")
    sha256_hash = hashlib.sha256(message).digest()

    # Concatenate path + SHA256 hash
    sign_message = url_path.encode("utf-8") + sha256_hash

    # Decode the secret from base64
    secret_bytes = base64.b64decode(api_secret)

    # HMAC-SHA512 signature
    signature = hmac.new(secret_bytes, sign_message, hashlib.sha512)

    # Return base64-encoded signature
    return base64.b64encode(signature.digest()).decode("utf-8")


def get_auth_headers(
    api_key: str,
    api_secret: str,
    url_path: str,
    nonce: str,
    post_data: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Get authentication headers for Kraken API request.

    Args:
        api_key: Kraken API key
        api_secret: Kraken API secret (base64-encoded)
        url_path: API endpoint path
        nonce: Unique nonce for this request
        post_data: POST data dictionary

    Returns:
        Dictionary with API-Key and API-Sign headers
    """
    signature = create_signature(api_secret, url_path, nonce, post_data)

    return {
        "API-Key": api_key,
        "API-Sign": signature,
    }


def validate_credentials(api_key: str, api_secret: str) -> bool:
    """
    Validate that Kraken credentials are properly formatted.

    Args:
        api_key: Kraken API key
        api_secret: Kraken API secret

    Returns:
        True if credentials appear valid

    Raises:
        ValueError: If credentials are invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise ValueError("Kraken API key is required")

    if not api_secret or not isinstance(api_secret, str):
        raise ValueError("Kraken API secret is required")

    # Verify secret is valid base64
    try:
        decoded = base64.b64decode(api_secret)
        if len(decoded) < 32:
            raise ValueError("Kraken API secret appears too short")
    except Exception as e:
        raise ValueError(f"Kraken API secret is not valid base64: {e}")

    return True
