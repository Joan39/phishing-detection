"""Validate external QPhish deployment dependencies without exposing secrets."""

from __future__ import annotations

import os
import smtplib
import socket
import ssl
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not configured.")
    return value


def check_database() -> None:
    url = require("DATABASE_URL")
    if not url.startswith(("postgresql://", "postgres://")):
        raise RuntimeError("DATABASE_URL must use PostgreSQL.")
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=5) as connection:
            connection.execute("SELECT 1").fetchone()
    except Exception as exc:
        raise RuntimeError(f"PostgreSQL connection failed: {exc}") from exc
    print("PostgreSQL: reachable")


def check_smtp() -> None:
    host = require("QPHISH_SMTP_HOST")
    port = int(os.getenv("QPHISH_SMTP_PORT", "587"))
    username = require("QPHISH_SMTP_USERNAME")
    require("QPHISH_SMTP_PASSWORD")
    require("QPHISH_SMTP_FROM")
    try:
        with smtplib.SMTP(host, port, timeout=10) as server:
            server.starttls(context=ssl.create_default_context())
            server.login(username, os.environ["QPHISH_SMTP_PASSWORD"])
    except Exception as exc:
        raise RuntimeError(f"SMTP connection/authentication failed: {exc}") from exc
    print("SMTP: reachable and authenticated")


def check_tls_files() -> None:
    certificate = ROOT / "deploy" / "certs" / "fullchain.pem"
    private_key = ROOT / "deploy" / "certs" / "privkey.pem"
    missing = [str(path) for path in (certificate, private_key) if not path.is_file()]
    if missing:
        raise RuntimeError(f"TLS files missing: {', '.join(missing)}")
    print("TLS certificates: present")


def check_configuration() -> None:
    secret = require("QPHISH_AUTH_SECRET")
    if len(secret) < 32 or secret.lower().startswith("replace-with"):
        raise RuntimeError("QPHISH_AUTH_SECRET is too short or still a placeholder.")
    hosts = require("QPHISH_ALLOWED_HOSTS")
    origins = require("QPHISH_FRONTEND_ORIGINS")
    if "example.com" in hosts or "example.com" in origins:
        raise RuntimeError("Production host/origin values are still placeholders.")
    print("Secrets and domains: configured")


def check_docker() -> None:
    import shutil
    import subprocess

    executable = shutil.which("docker")
    if not executable:
        raise RuntimeError("Docker is not installed or is not on PATH.")
    result = subprocess.run(
        [executable, "compose", "config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Docker Compose validation failed: {result.stderr.strip()}")
    print("Docker Compose: configuration valid")


def main() -> int:
    checks = (
        ("configuration", check_configuration),
        ("database", check_database),
        ("smtp", check_smtp),
        ("tls", check_tls_files),
        ("docker", check_docker),
    )
    failures = 0
    for name, check in checks:
        try:
            check()
        except (RuntimeError, ValueError, OSError) as exc:
            failures += 1
            print(f"{name}: FAILED — {exc}", file=sys.stderr)
    if failures:
        print(f"Preflight failed with {failures} check(s).", file=sys.stderr)
        return 1
    print("All external deployment checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
