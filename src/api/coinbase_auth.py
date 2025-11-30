"""
Custom JWT authentication for Coinbase API with Ed25519 support.

The official coinbase-advanced-py library doesn't support Ed25519 keys yet,
so we implement our own authentication.
"""

import base64
import json
import secrets
import time
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import jwt
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def build_jwt(
    api_key: str,
    api_secret: str,
    method: str = "GET",
    path: str = "",
) -> str:
    """
    Build a JWT token for Coinbase API authentication.

    Supports Ed25519 keys (base64 encoded) used by new CDP API keys.

    Args:
        api_key: The API key ID (UUID format for Ed25519)
        api_secret: The private key (base64 encoded for Ed25519)
        method: HTTP method (GET, POST, etc.)
        path: API endpoint path

    Returns:
        Signed JWT token
    """
    # Check if this is an Ed25519 key (base64, not PEM)
    if not api_secret.startswith("-----BEGIN"):
        # Ed25519 key - base64 encoded
        return _build_jwt_ed25519(api_key, api_secret, method, path)
    else:
        # ECDSA key - PEM format (use existing library)
        raise ValueError("ECDSA keys should use the standard library authentication")


def _build_jwt_ed25519(
    api_key: str,
    api_secret: str,
    method: str,
    path: str,
) -> str:
    """Build JWT using Ed25519 signature."""
    # Decode the base64 private key
    # Ed25519 keys from CDP are 64 bytes: 32 byte seed + 32 byte public key
    key_bytes = base64.b64decode(api_secret)

    if len(key_bytes) == 64:
        # Use first 32 bytes as seed
        seed = key_bytes[:32]
    elif len(key_bytes) == 32:
        seed = key_bytes
    else:
        raise ValueError(f"Invalid Ed25519 key length: {len(key_bytes)} bytes")

    # Create Ed25519 private key from seed
    private_key = Ed25519PrivateKey.from_private_bytes(seed)

    # Build JWT claims
    now = int(time.time())
    nonce = secrets.token_hex(8)

    # Build URI claim
    uri = f"{method} api.coinbase.com{path}"

    payload = {
        "sub": api_key,
        "iss": "cdp",
        "aud": ["cdp_service"],
        "nbf": now,
        "exp": now + 60,  # 60 second expiry
        "uris": [uri],
    }

    headers = {
        "alg": "EdDSA",
        "typ": "JWT",
        "kid": api_key,
        "nonce": nonce,
    }

    # Sign with Ed25519
    token = jwt.encode(
        payload,
        private_key,
        algorithm="EdDSA",
        headers=headers,
    )

    return token


def load_key_file(key_file: Union[str, Path]) -> tuple[str, str]:
    """
    Load API credentials from a CDP key file.

    Args:
        key_file: Path to the JSON key file

    Returns:
        Tuple of (api_key, api_secret)
    """
    key_path = Path(key_file)
    if not key_path.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")

    with open(key_path) as f:
        key_data = json.load(f)

    # Handle different key file formats
    # New Ed25519 format: {"id": "...", "privateKey": "base64..."}
    # Old ECDSA format: {"name": "organizations/.../apiKeys/...", "privateKey": "-----BEGIN..."}
    api_key = key_data.get("name") or key_data.get("id")
    api_secret = key_data.get("privateKey")

    if not api_key or not api_secret:
        raise ValueError("Invalid key file format - missing id/name or privateKey")

    return api_key, api_secret
