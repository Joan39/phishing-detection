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
# without them the system flags any unfamiliar path as phishing. Mixed TLD
# families, including hosts whose real traffic still arrives over plain HTTP.
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
    "pytorch.org",
    "www.tensorflow.org",
    "scikit-learn.org",
    "pandas.pydata.org",
    "numpy.org",
    "www.docker.com",
    "code.visualstudio.com",
    "www.eclipse.org",
    "www.kernel.org",
    "lists.debian.org",
    "www.drupal.org",
    "www.joomla.org",
    "moodle.org",
    "www.wikidata.org",
    "commons.wikimedia.org",
    "www.nature.com",
    "www.cell.com",
    "journals.aps.org",
    "link.aps.org",
    "www.sciencedirect.com",
    "pmc.ncbi.nlm.nih.gov",
    "www.cdc.gov",
    "www.who.int",
    "ec.europa.eu",
    "www.bund.de",
    "data.europa.eu",
    "www.ons.gov.uk",
    "www.abs.gov.au",
    "www.stat.go.jp",
    "www.insee.fr",
    "www.scb.se",
    "www.leibniz-gemeinschaft.de",
    "www.uni-leiden.nl",
    "www.kuleuven.be",
    "www.usp.br",
    "www.unam.mx",
    "www.msu.ru",
    "httpwg.org",
    "datatracker.ietf.org",
    "www.fastly.com",
    "blog.cloudflare.com",
    "www.stats.gov.cn",
    "www.osaka-u.ac.jp",
    "cms.gov",
    "www.epa.gov",
    "www.fda.gov",
    "www.uspto.gov",
    "www.govinfo.gov",
    "www.gutenberg.org",
    "www.librivox.org",
    "openlibrary.org",
    "www.mcgill.ca",
    "www.tandf.co.uk",
    "brage.unit.no",
)

# Path shapes that ordinary sites really use. Digit-heavy identifiers, hyphenated
# slugs and query strings are deliberate: they are the legitimate shapes the
# corpus cannot teach, because its legitimate rows are bare domains.
GENERAL_LEGITIMATE_PATHS = (
    "",
    "/",
    "/index.html",
    "/about",
    "/about-us.html",
    "/contact",
    "/support/faq",
    "/docs/getting-started.pdf",
    "/products/item-123",
    "/abs/2301.05427",
    "/doi/10.1103/PhysRevLett.121.070501",
    "/pmc/articles/PMC8421310/",
    "/questions/12345678/how-to-parse-a-url",
    "/blog/2026/03/14/how-models-see-urls",
    "/news/2026/technology/quantum-machine-learning",
    "/guidance/report-a-phishing-email",
    "/project/requests/",
    "/package/django/",
    "/course/quantum-computing-i/",
    "/statistics/2026/quarter-1.html",
    "/record/4421877",
    "/files/annual-report-2025.pdf",
    "/catalog/record?id=48213",
    "/item?id=12345678",
    "/search?q=quantum+computing&page=2",
    "/api/v2/status.json?format=compact",
    "/en/latest/configuration.html",
    "/de/startseite.html",
    "/user/84213/profile",
    "/solution/1000234.html",
    "/news/2026/03/editors-pick-on-renewable-power",
    "/science-and-technology/2026-03-14/a-field-guide-to-proteins",
    "/technology/2026/mar/14/open-source-tools-worth-knowing",
    "/features/long-read/the-last-good-news-story",
    "/help/get-started/installation-troubleshooting",
    "/en/stable/guides/deployment-checklist.html",
    "/press-room/releases/2026/04/quarterly-results-summary",
    "/documents/2025/annual-sustainability-report.pdf",
    "/guides/beginners/how-to-read-a-credential-prompt",
    "/news-analysis/2026/04/what-the-new-rule-means-for-you",
    "/articles/research/measurement-of-quantum-noise-2025.pdf",
    "/learning/path-2/module-7/lesson-3.html",
    "/news-events/news-releases/2026/04/research-brief-118",
)

# Each legitimate counter-example is repeated this many times in production
# training. The level comes from scripts/legit_negative_experiment.py: 10
# repetitions of the broad host/path set recovered most real-world legitimate
# traffic without letting HTTP lookalike hosts slip under the threshold, while
# 20 repetitions of the same shapes cost a paypa1-style lookalike.
HARD_NEGATIVE_REPLICAS = 10

# Bare HTTP origins are weighted harder than the rest of the corpus because the
# dataset has no HTTP legitimate rows at all. scripts/legit_negative_experiment.py
# measured the tradeoff: 60 replicas recovered real-world legitimate traffic
# while holding the held-out fold accuracy and the phishing battery flat; 200
# replicas bought a little more legitimate traffic but cost fold accuracy.
BARE_HTTP_REPLICAS = 60


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


def general_legitimate_training_urls(
    http_paths_per_host: int | None = None,
) -> list[str]:
    """Return legitimate counter-examples for ordinary non-brand sites.

    ``http_paths_per_host`` controls how many path shapes also get a plain-HTTP
    variant for every host; the corpus has no HTTP legitimate rows at all, so
    this is the only place the models can learn that an unencrypted origin is
    ordinary for some sites rather than proof of phishing.
    """
    if http_paths_per_host is None:
        http_paths_per_host = len(GENERAL_LEGITIMATE_PATHS)
    urls: set[str] = set()
    for host in GENERAL_LEGITIMATE_HOSTS:
        for path in GENERAL_LEGITIMATE_PATHS:
            urls.add(f"https://{host}{path}")
    # Plain HTTP is still ordinary traffic, so every host contributes a bare
    # HTTP origin plus the requested number of path shapes.
    for host in GENERAL_LEGITIMATE_HOSTS:
        urls.update({f"http://{host}", f"http://{host}/"})
        for path in GENERAL_LEGITIMATE_PATHS[: max(http_paths_per_host, 0)]:
            urls.add(f"http://{host}{path}")
    return sorted(urls)


def legitimate_training_urls(
    include_general_hosts: bool = True,
    http_paths_per_host: int | None = None,
) -> list[str]:
    """Return the full legitimate hard-negative URL set used for production."""
    urls = set(trusted_root_training_urls())
    if include_general_hosts:
        urls.update(general_legitimate_training_urls(http_paths_per_host))
    return sorted(urls)


def bare_http_origin_training_urls() -> list[str]:
    """Return the scheme-only counter-examples: ``http://host`` and ``http://host/``.

    These are the shortest legitimate rows the corpus can offer, so a plain-HTTP
    origin on an unfamiliar host otherwise has nothing to anchor on. They are
    weighted separately from the wider corpus because their count, not their
    shape, decides how much HTTP the models tolerate.
    """
    return sorted(
        {f"http://{host}{suffix}" for host in GENERAL_LEGITIMATE_HOSTS for suffix in ("", "/")}
    )


def hard_negative_blocks(
    include_general_hosts: bool = True,
    http_paths_per_host: int | None = None,
) -> list[tuple[list[str], int]]:
    """Return ``(urls, replicas)`` blocks in the exact production weighting.

    Training and evaluation share this function so a chosen configuration cannot
    drift from what ``build_production_training_data`` actually fits.
    """
    blocks = [
        (
            legitimate_training_urls(include_general_hosts, http_paths_per_host),
            HARD_NEGATIVE_REPLICAS,
        )
    ]
    if include_general_hosts and BARE_HTTP_REPLICAS:
        blocks.append((bare_http_origin_training_urls(), BARE_HTTP_REPLICAS))
    return blocks
