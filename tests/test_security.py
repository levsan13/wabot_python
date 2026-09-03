"""Webhook signature validation."""

from __future__ import annotations

from app.whatsapp.security import compute_signature, verify_signature

SECRET = "segredo-do-app"
BODY = b'{"object":"whatsapp_business_account"}'


def test_valid_signature():
    assert verify_signature(SECRET, BODY, compute_signature(SECRET, BODY))


def test_invalid_signature():
    assert not verify_signature(SECRET, BODY, "sha256=deadbeef")


def test_missing_header_is_refused():
    assert not verify_signature(SECRET, BODY, None)


def test_tampered_body_is_refused():
    """One extra byte changes the digest — this is the whole point of the check."""
    signature = compute_signature(SECRET, BODY)
    assert not verify_signature(SECRET, BODY + b" ", signature)


def test_no_secret_skips_the_check():
    # Development mode: with no WA_APP_SECRET the validation is bypassed.
    assert verify_signature("", BODY, None)
