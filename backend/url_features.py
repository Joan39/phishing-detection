"""Extract URL features for the prototype inference path.

These are transparent lexical definitions, not a claim to reproduce every
private feature-engineering detail used to create PhiUSIIL. They must be
calibrated against the dataset's original feature generator before deployment.
"""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

import pandas as pd

from .train_models import URL_DERIVED_FEATURES
from .domain_policy import is_trusted_root_url


OBFUSCATED_TOKEN = re.compile(r"%[0-9A-Fa-f]{2}")
OTHER_SPECIAL_CHARACTERS = set("@!#$%^*()[]{}'\";,<>|\\")


def extract_url_features(url: str) -> pd.DataFrame:
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
    return pd.DataFrame([ordered])
