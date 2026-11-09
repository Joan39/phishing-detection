"""Extract URL features for the prototype inference path.

These are transparent lexical definitions, not a claim to reproduce every
private feature-engineering detail used to create PhiUSIIL. They must be
calibrated against the dataset's original feature generator before deployment.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from .train_models import URL_DERIVED_FEATURES
from .domain_policy import is_trusted_domain_url, is_trusted_root_url


OBFUSCATED_TOKEN = re.compile(r"%[0-9A-Fa-f]{2}")
OTHER_SPECIAL_CHARACTERS = set("@!#$%^*()[]{}'\";,<>|\\")

# Probability ceiling applied to a URL on a configured trusted domain.
TRUSTED_DOMAIN_CAP = 0.25

# Query parameters whose value is a URL the page sends the visitor to. These are
# the names used by the open-redirect payloads observed evading the brand hosts.
REDIRECT_TARGET_PARAMETERS = frozenset(
    {
        "url",
        "u",
        "uri",
        "next",
        "target",
        "dest",
        "destination",
        "goto",
        "redirect",
        "redirect_url",
        "redirect_uri",
        "redirect_to",
        "return",
        "return_url",
        "returnurl",
        "continue",
    }
)


def _registrable_domain(hostname: str) -> str:
    """Naive eTLD+1: the last two labels, enough to tell one brand from another.

    No public-suffix list is consulted, so ``amazon.com`` following
    ``amazon.co.uk`` is treated as a foreign jump. That direction is safe here:
    the result is that the link is scored on its own content rather than being
    trusted because of the visible brand host.
    """
    labels = [label for label in hostname.split(".") if label]
    return ".".join(labels[-2:]) if len(labels) >= 2 else hostname.lower()


def _absolute_target(value: str) -> str | None:
    candidate = value.strip()
    if candidate.startswith("//"):
        candidate = "https:" + candidate
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def redirect_targets(url: str) -> list[str]:
    """Absolute URLs embedded in redirect parameters of ``url``.

    Only foreign targets are returned: a brand that links to itself is doing
    ordinary navigation, while a brand whose entire purpose is a jump to
    ``refund-claim-irs-com`` is the payload the detector has never seen. Query
    pairs and query strings inside the fragment are both percent-decoded, which
    is what a browser actually follows.
    """
    parsed = urlsplit((url or "").strip())
    pairs = parse_qs(parsed.query, keep_blank_values=False)
    fragment = parsed.fragment
    if "=" in fragment:
        pairs = {
            **pairs,
            **parse_qs(
                fragment.partition("?")[2] if "?" in fragment else fragment,
                keep_blank_values=False,
            ),
        }
    origin = _registrable_domain((parsed.hostname or "").lower())
    targets: list[str] = []
    for key, values in pairs.items():
        if key.strip().lower() not in REDIRECT_TARGET_PARAMETERS:
            continue
        for value in values:
            candidate = _absolute_target(value)
            if candidate is None:
                continue
            host = (urlsplit(candidate).hostname or "").lower()
            if _registrable_domain(host) != origin and candidate not in targets:
                targets.append(candidate)
    return targets


def trusted_calibration_applies(url: str, features: pd.DataFrame) -> bool:
    """Return whether the trusted-domain probability cap may be applied.

    A trusted hostname alone is not enough: an IP host, percent-encoded
    obfuscation, unusual special characters or excessive subdomain depth mean
    the URL is not behaving like the real brand, so it is scored normally.
    """
    return (
        is_trusted_domain_url(url)
        and not bool(features["IsDomainIP"].iloc[0])
        and not bool(features["HasObfuscation"].iloc[0])
        # Fragments are never sent to the server, so only pre-fragment
        # special characters disqualify a trusted host from calibration.
        and sum(
            character in OTHER_SPECIAL_CHARACTERS
            for character in url.split("#", 1)[0]
        )
        == 0
        and int(features["NoOfSubDomain"].iloc[0]) <= 2
        # A brand page that only carries the visitor to another domain is not
        # the brand's own content, so it earns no calibration.
        and not redirect_targets(url)
    )


def extract_url_features(url: str) -> pd.DataFrame:
    return pd.DataFrame([url_feature_values(url)])


def extract_url_features_frame(urls) -> pd.DataFrame:
    """Extract features for many URLs into one frame.

    Building one row frame per URL and concatenating them is quadratic and
    dominates retraining time on the full corpus, so batch callers get a single
    frame instead.
    """
    return pd.DataFrame([url_feature_values(url) for url in urls])


def url_feature_values(url: str) -> dict[str, float]:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    raw_url = url.strip()
    candidate = raw_url if "://" in raw_url else "//" + raw_url
    try:
        parsed = urlsplit(candidate)
        hostname = parsed.hostname or ""
        parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid host or port.") from exc
    if any(character.isspace() for character in hostname):
        raise ValueError("URL contains whitespace in its hostname.")
    length = len(raw_url)
    letters = sum(character.isalpha() for character in raw_url)
    digits = sum(character.isdigit() for character in raw_url)
    obfuscated_tokens = OBFUSCATED_TOKEN.findall(raw_url)
    special_characters = sum(
        character in OTHER_SPECIAL_CHARACTERS for character in raw_url
    )
    host_parts = [part for part in hostname.split(".") if part]
    tld_length = len(host_parts[-1]) if host_parts else 0

    try:
        is_domain_ip = int(
            bool(hostname) and ipaddress.ip_address(hostname) is not None
        )
    except ValueError:
        is_domain_ip = 0

    values = {
        "URLLength": length,
        "DomainLength": len(hostname),
        "IsDomainIP": is_domain_ip,
        "TLDLength": tld_length,
        "NoOfSubDomain": max(len(host_parts) - 2, 0),
        "HasObfuscation": int(bool(obfuscated_tokens)),
        "NoOfObfuscatedChar": len(obfuscated_tokens) * 3,
        "ObfuscationRatio": len(obfuscated_tokens) * 3 / length,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / length,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / length,
        "NoOfEqualsInURL": raw_url.count("="),
        "NoOfQMarkInURL": raw_url.count("?"),
        "NoOfAmpersandInURL": raw_url.count("&"),
        "NoOfOtherSpecialCharsInURL": special_characters,
        "SpacialCharRatioInURL": special_characters / length,
        "IsHTTPS": int(parsed.scheme.lower() == "https"),
        "IsTrustedRoot": int(is_trusted_root_url(raw_url)),
    }
    ordered = {
        feature: values[feature]
        for feature in URL_DERIVED_FEATURES
        if feature in values
    }
    return ordered
