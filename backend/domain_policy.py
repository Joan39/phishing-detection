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

# Realistic legitimate URL shapes. The PhiUSIIL legitimate rows are almost all
# bare domains, so models trained on the dataset alone label any long URL with
# paths, queries, hyphens or digits as phishing. These templates give the
# production models counter-examples that match real site traffic.
LEGITIMATE_URL_TEMPLATES = (
    "",
    "/",
    "/search?q=quantum+computing&hl=en",
    "/search?q=phishing+report&client=firefox-b#/result-3",
    "/news/2026/technology/neural-networks-explained",
    "/blog/post/how-to-spot-fake-login-pages-2",
    "/products/item-12345?color=blue&size=lg",
    "/docs/getting-started/install-on-linux.html",
    "/en-us/support/faq/contact-form",
    "/wiki/Phishing",
    "/wiki/Amanita_muscaria",
    "/category/books/history?page=2",
    "/user/profile?id=42",
    "/api/v1/status.json",
    "/files/2024/annual-report.pdf",
    "/about/contact-us",
    "/results?search_query=awareness+video&sp=1",
    "/gp/new-releases/books/1/ref=zg_b_tab_1",
    "/blogs/security/protecting-accounts-from-phishing",
    "/en-us/102237",
    "/store/b/surface?icid=nav-top",
    "/V1/ServiceLogin?hl=en&continue=home",
    "/login?redirect=%2Fdashboard",
    "/c/2025/03/14/tech-ai-models-update/",
)

HARD_NEGATIVE_HOSTS = (
    "www.google.com",
    "accounts.google.com",
    "docs.google.com",
    "www.wikipedia.org",
    "en.wikipedia.org",
    "github.com",
    "status.github.com",
    "docs.github.com",
    "www.microsoft.com",
    "support.microsoft.com",
    "www.amazon.com",
    "aws.amazon.com",
    "www.youtube.com",
    "support.apple.com",
    "www.example.com",
)

# Ordinary legitimate hosts: real organisations that are not brand-targeted
# enough to belong in the trusted list, but whose URLs users do type. The
# dataset's legitimate rows never look like these (bare domains, all HTTPS), so
# without them the system flags any unfamiliar path as phishing.
GENERAL_LEGITIMATE_HOSTS = (
    "www.mozilla.org",
    "developer.mozilla.org",
    "www.chromium.org",
    "www.apache.org",
    "nginx.org",
    "redis.io",
    "kubernetes.io",
    "go.dev",
    "www.rust-lang.org",
    "www.php.net",
    "www.npmjs.com",
    "www.w3.org",
    "www.openssl.org",
    "curl.se",
    "www.nasa.gov",
    "www.noaa.gov",
    "web.mit.edu",
    "www.stanford.edu",
    "www.harvard.edu",
    "www.cam.ac.uk",
    "ethz.ch",
    "www.kaggle.com",
    "www.coursera.org",
    "www.edx.org",
    "www.icann.org",
    "www.iana.org",
    "archive.org",
    "www.openstreetmap.org",
    "www.debian.org",
    "ubuntu.com",
    "fedoraproject.org",
)

# Short, ordinary paths. Kept short because the shapes users actually reach over
# plain HTTP are simple; long hyphen-heavy paths look like attack hosts.
GENERAL_LEGITIMATE_PATHS = (
    "",
    "/",
    "/index.html",
    "/about",
    "/about-us.html",
    "/contact",
    "/docs/getting-started.pdf",
    "/news/2026/article",
    "/products/item-123",
    "/support/faq",
)

# Each template/host example is repeated this many times in production
# training so the counter-examples outweigh the bare-domain majority.
HARD_NEGATIVE_REPLICAS = 250


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
    """Return legitimate hard negatives for production training.

    Real-world legitimate URLs routinely carry paths, query parameters and
    subdomains, while the dataset's legitimate rows are almost bare domains.
    Without these hard negatives the models learn "any parameter means
    phishing" and false-positive on ordinary site traffic. Rows are expanded
    from hosts x templates; build_production_training_data applies
    HARD_NEGATIVE_REPLICAS as a class weight.
    """
    domains = sorted(trusted_root_domains())
    urls = {f"https://{domain}" for domain in domains}
    urls.update(
        {
            "https://accounts.google.com",
            "https://github.com/login",
            "https://www.microsoft.com/en-us",
            "https://www.wikipedia.org/wiki/Main_Page",
            "https://www.youtube.com",
            "https://example.com",
        }
    )
    for host in HARD_NEGATIVE_HOSTS:
        for template in LEGITIMATE_URL_TEMPLATES:
            urls.add(f"https://{host}{template}")
    for host in HARD_NEGATIVE_HOSTS[:4]:
        for template in LEGITIMATE_URL_TEMPLATES[:8]:
            urls.add(f"http://{host}{template}")
    return sorted(urls)


def general_legitimate_training_urls() -> list[str]:
    """Return legitimate counter-examples for ordinary non-brand sites."""
    urls: set[str] = set()
    for host in GENERAL_LEGITIMATE_HOSTS:
        for path in GENERAL_LEGITIMATE_PATHS:
            urls.add(f"https://{host}{path}")
    for host in GENERAL_LEGITIMATE_HOSTS[: len(GENERAL_LEGITIMATE_HOSTS) // 2]:
        for path in GENERAL_LEGITIMATE_PATHS[:6]:
            urls.add(f"http://{host}{path}")
    return sorted(urls)


def legitimate_training_urls(include_general_hosts: bool = True) -> list[str]:
    """Return the full legitimate hard-negative URL set used for production."""
    urls = set(trusted_root_training_urls())
    if include_general_hosts:
        urls.update(general_legitimate_training_urls())
    return sorted(urls)
