"""Splitting long answers to fit the Cloud API limits."""

from __future__ import annotations

from app.whatsapp.client import SAFE_CHUNK, TEXT_LIMIT, split_text


def test_short_text_is_left_alone():
    assert split_text("oi") == ["oi"]


def test_blank_text_yields_nothing():
    assert split_text("   ") == []


def test_chunks_respect_the_limit():
    text = "palavra " * 3000
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= SAFE_CHUNK for c in chunks)
    assert all(len(c) <= TEXT_LIMIT for c in chunks)


def test_split_prefers_paragraph_boundaries():
    text = ("a" * 2000) + "\n\n" + ("b" * 2000)
    chunks = split_text(text, limit=2500)
    assert chunks[0] == "a" * 2000
    assert chunks[1] == "b" * 2000


def test_nothing_is_lost():
    text = "linha\n" * 2000
    joined = "".join(split_text(text)).replace("\n", "")
    assert joined == text.replace("\n", "")
