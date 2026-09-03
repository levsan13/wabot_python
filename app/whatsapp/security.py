"""Validation of the X-Hub-Signature-256 header Meta sends with each webhook."""

from __future__ import annotations

import hashlib
import hmac
import logging

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "x-hub-signature-256"


def compute_signature(app_secret: str, raw_body: bytes) -> str:
    """HMAC-SHA256 of the raw request body, in Meta's `sha256=<hex>` shape."""
    digest = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(app_secret: str, raw_body: bytes, header_value: str | None) -> bool:
    """Compare the signature of the raw body against the received header.

    Must run on the RAW bytes: re-serializing the parsed JSON changes the
    digest. With no WA_APP_SECRET the check is skipped, which is handy in local
    development — but in production leaving it empty lets anyone who finds the
    URL drive the bot.
    """
    if not app_secret:
        logger.debug("WA_APP_SECRET empty — signature validation disabled")
        return True
    if not header_value:
        logger.warning("Webhook arrived without the %s header", SIGNATURE_HEADER)
        return False
    # compare_digest keeps the comparison constant-time.
    return hmac.compare_digest(compute_signature(app_secret, raw_body), header_value.strip())
