"""Small, explicit URL policy helpers shared by training and inference."""

from __future__ import annotations

import os
from urllib.parse import urlsplit


DEFAULT_TRUSTED_ROOT_DOMAINS = frozenset(
    {
        "amazon.com",
        "apple.com",
        "github.com",
        "google.com",
        "microsoft.com",
        "wikipedia.org",
        "youtube.com",
    }
)


def trusted_root_domains() -> frozenset[str]:
    configured = os.getenv("QPHISH_TRUSTED_ROOT_DOMAINS")
    if configured is None:
        return DEFAULT_TRUSTED_ROOT_DOMAINS
    return frozenset(
        domain.strip().lower().removeprefix("www.")
        for domain in configured.split(",")
        if domain.strip()
    )


def is_trusted_root_url(url: str) -> bool:
    parsed = urlsplit(url.strip())
    hostname = (parsed.hostname or "").lower().rstrip(".").removeprefix("www.")
    return (
        hostname in trusted_root_domains()
        and parsed.path in {"", "/"}
        and not parsed.query
        and not parsed.fragment
        and parsed.port in {None, 80, 443}
    )


def is_trusted_domain_url(url: str) -> bool:
    """Return whether the URL host belongs to a configured trusted domain."""
    parsed = urlsplit(url.strip())
    hostname = (parsed.hostname or "").lower().rstrip(".").removeprefix("www.")
    return (
        parsed.scheme.lower() in {"http", "https"}
        and bool(hostname)
        and any(
            hostname == domain or hostname.endswith(f".{domain}")
            for domain in trusted_root_domains()
        )
        and parsed.port in {None, 80, 443}
    )


def trusted_root_training_urls() -> list[str]:
    """Return clean root and canonical-page examples for production training."""
    domains = sorted(trusted_root_domains())
    canonical_urls = {
        "https://accounts.google.com",
        "https://github.com/login",
        "https://www.microsoft.com/en-us",
        "https://www.wikipedia.org/wiki/Main_Page",
        "https://www.youtube.com",
        "https://example.com",
    }
    return [f"https://{domain}" for domain in domains] + sorted(canonical_urls)
