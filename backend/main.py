"""QPhish API for the URL-only saved model inference path."""

from __future__ import annotations

import json
import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
import smtplib
from email.message import EmailMessage
import psycopg
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from .train_models import (
    DEFAULT_DATASET,
    build_feature_frame,
    build_production_training_data,
    make_baseline,
    make_ensemble,
)
from .domain_policy import is_trusted_domain_url
from .quantum_live import (
    adapt_live_model,
    predict_quantum_probability,
)
from .url_features import OTHER_SPECIAL_CHARACTERS, extract_url_features


ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = Path(os.getenv("QPHISH_DATABASE_FILE", ROOT / "results" / "qphish.db"))
DATABASE_URL = os.getenv("DATABASE_URL")
AUTH_SECRET = os.getenv("QPHISH_AUTH_SECRET")
if not AUTH_SECRET:
    AUTH_SECRET_FILE = Path(
        os.getenv("QPHISH_AUTH_SECRET_FILE", ROOT / "results" / "auth.secret")
    )
    AUTH_SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    if AUTH_SECRET_FILE.exists():
        AUTH_SECRET = AUTH_SECRET_FILE.read_text(encoding="utf-8").strip()
    else:
        AUTH_SECRET = secrets.token_urlsafe(32)
        AUTH_SECRET_FILE.write_text(AUTH_SECRET, encoding="utf-8")
        logging.getLogger(__name__).warning(
            "QPHISH_AUTH_SECRET is unset; generated a persistent local signing key."
        )
    if len(AUTH_SECRET) < 32:
        raise RuntimeError("QPHISH_AUTH_SECRET must contain at least 32 characters.")
SESSION_TTL_SECONDS = int(os.getenv("QPHISH_SESSION_TTL_SECONDS", "28800"))
EMAIL_TOKEN_TTL_SECONDS = int(os.getenv("QPHISH_EMAIL_TOKEN_TTL_SECONDS", "86400"))
PUBLIC_API_URL = os.getenv("QPHISH_PUBLIC_API_URL", "http://localhost:8000")
SMTP_HOST = os.getenv("QPHISH_SMTP_HOST")
SMTP_PORT = int(os.getenv("QPHISH_SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("QPHISH_SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("QPHISH_SMTP_PASSWORD")
SMTP_FROM = os.getenv("QPHISH_SMTP_FROM")
REQUIRE_EMAIL_VERIFICATION = os.getenv(
    "QPHISH_REQUIRE_EMAIL_VERIFICATION", "false"
).lower() == "true"
REQUIRE_HTTPS = os.getenv("QPHISH_REQUIRE_HTTPS", "false").lower() == "true"
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("QPHISH_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("QPHISH_RATE_LIMIT_MAX_REQUESTS", "60"))
MODEL_DIR = Path(os.getenv("QPHISH_MODEL_DIR", ROOT / "models" / "url"))
RL_STATE_FILE = Path(
    os.getenv("QPHISH_RL_STATE_FILE", ROOT / "results" / "rl_state.json")
)
QUANTUM_MODEL_FILE = Path(
    os.getenv("QPHISH_QUANTUM_MODEL_FILE", ROOT / "models" / "quantum_live.joblib")
)
QUANTUM_WEIGHT = float(os.getenv("QPHISH_QUANTUM_WEIGHT", "0.25"))
# Ensemble weights reflect held-out performance: the soft-voting ensemble
# dominates single-model logistic regression on every metric, so it carries
# most of the classical probability.
CLASSICAL_WEIGHTS = {"A": 0.35, "B": 0.65}
MODEL_LOCK = threading.Lock()
ADAPTATION_LOCK = threading.Lock()
ADAPTATION_RUNNING = False
RETRAIN_RUNNING = False
logger = logging.getLogger("qphish")
REQUEST_COUNT = Counter(
    "qphish_http_requests_total", "HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "qphish_http_request_duration_seconds", "HTTP request latency", ["method", "path"]
)
SCAN_COUNT = Counter("qphish_scans_total", "URL scans", ["verdict"])
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_BUCKETS: dict[str, list[float]] = {}
FRONTEND_ORIGINS = os.getenv(
    "QPHISH_FRONTEND_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")
ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("QPHISH_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if host.strip()
]

app = FastAPI(title="QPhish API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in FRONTEND_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    forwarded_proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if REQUIRE_HTTPS and request.url.path.startswith("/api/"):
        if forwarded_proto != "https":
            return Response(
                content=json.dumps({"detail": "HTTPS is required."}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json",
            )
    if request.url.path.startswith("/api/"):
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with RATE_LIMIT_LOCK:
            bucket = [
                timestamp
                for timestamp in RATE_LIMIT_BUCKETS.get(client, [])
                if now - timestamp < RATE_LIMIT_WINDOW_SECONDS
            ]
            if len(bucket) >= RATE_LIMIT_MAX_REQUESTS:
                return Response(
                    content=json.dumps({"detail": "Too many requests."}),
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    media_type="application/json",
                    headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
                )
            bucket.append(now)
            RATE_LIMIT_BUCKETS[client] = bucket
        started = time.perf_counter()
        response = await call_next(request)
        REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(
            time.perf_counter() - started
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if forwarded_proto == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response
    return await call_next(request)


class _PostgresRow(dict):
    def __getitem__(self, key):
        if isinstance(key, int):
            return tuple(self.values())[key]
        return super().__getitem__(key)


class _PostgresConnection:
    def __init__(self, connection):
        self._connection = connection

    def __enter__(self):
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._connection.__exit__(exc_type, exc_value, traceback)

    def execute(self, query, parameters=()):
        return self._connection.execute(query.replace("?", "%s"), parameters)


def _postgres_row_factory(cursor):
    columns = [column.name for column in cursor.description]
    return lambda values: _PostgresRow(zip(columns, values))


def _db():
    if DATABASE_URL:
        connection = psycopg.connect(
            DATABASE_URL, row_factory=_postgres_row_factory
        )
        return _PostgresConnection(connection)
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _init_db() -> None:
    if DATABASE_URL:
        with _db() as connection:
            schema = """
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    email_verified BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_id TEXT PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    expires_at BIGINT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scans (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    scan_url TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    probability DOUBLE PRECISION NOT NULL,
                    decision_threshold DOUBLE PRECISION NOT NULL,
                    model_scores TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id BIGSERIAL PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    scan_url TEXT NOT NULL,
                    predicted_verdict TEXT NOT NULL,
                    analyst_verdict TEXT NOT NULL,
                    probability DOUBLE PRECISION,
                    decision_threshold DOUBLE PRECISION,
                    comment TEXT,
                    reward DOUBLE PRECISION NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics (
                    key TEXT PRIMARY KEY,
                    value DOUBLE PRECISION NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS email_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id BIGINT NOT NULL REFERENCES users(id),
                    purpose TEXT NOT NULL,
                    expires_at BIGINT NOT NULL,
                    used_at TEXT
                );
                """
            for statement in schema.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute(
                "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                "email_verified BOOLEAN NOT NULL DEFAULT FALSE"
            )
        return
    with _db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                email_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                expires_at INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                scan_url TEXT NOT NULL,
                verdict TEXT NOT NULL,
                probability REAL NOT NULL,
                decision_threshold REAL NOT NULL,
                model_scores TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                scan_url TEXT NOT NULL,
                predicted_verdict TEXT NOT NULL,
                analyst_verdict TEXT NOT NULL,
                probability REAL,
                decision_threshold REAL,
                comment TEXT,
                reward REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metrics (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS email_tokens (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            purpose TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            used_at TEXT
            );
            """
        )
        try:
            connection.execute(
                "ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


@app.on_event("startup")
def startup() -> None:
    _init_db()


def _password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        salt_text, digest_text = encoded.split("$", 1)
        expected = base64.urlsafe_b64decode(digest_text.encode())
        actual = base64.urlsafe_b64decode(_password_hash(password, base64.urlsafe_b64decode(salt_text.encode())).split("$", 1)[1])
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _email_token(user_id: int, purpose: str) -> str:
    raw_token = secrets.token_urlsafe(48)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    with _db() as connection:
        connection.execute(
            "INSERT INTO email_tokens(token_hash,user_id,purpose,expires_at) VALUES(?,?,?,?)",
            (
                token_hash,
                user_id,
                purpose,
                int(time.time()) + EMAIL_TOKEN_TTL_SECONDS,
            ),
        )
    return raw_token


def _send_email(recipient: str, subject: str, body: str) -> None:
    if not all((SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM)):
        raise RuntimeError("SMTP settings are required for account email flows.")
    message = EmailMessage()
    message["From"] = SMTP_FROM
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(message)


def _token(user_id: int) -> str:
    expires = int(time.time()) + SESSION_TTL_SECONDS
    token_id = secrets.token_urlsafe(24)
    payload = f"{user_id}.{expires}.{token_id}"
    signature = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    with _db() as connection:
        connection.execute(
            "INSERT INTO sessions(token_id,user_id,expires_at,created_at) VALUES(?,?,?,?)",
            (token_id, user_id, expires, datetime.now(timezone.utc).isoformat()),
        )
    return f"{payload}.{signature}"


def _revoke_token(token: str) -> None:
    parts = token.split(".")
    if len(parts) == 4:
        with _db() as connection:
            connection.execute("DELETE FROM sessions WHERE token_id=?", (parts[2],))


def _current_user(authorization: str | None = Header(default=None)) -> sqlite3.Row:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication is required.")
    parts = authorization[7:].split(".")
    if len(parts) != 4:
        raise HTTPException(status_code=401, detail="Invalid session.")
    user_id, expires, token_id, signature = parts
    payload = ".".join(parts[:3])
    expected = hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(status_code=401, detail="Invalid session.")
    try:
        if int(expires) < int(time.time()):
            raise HTTPException(status_code=401, detail="Session expired.")
        with _db() as connection:
            row = connection.execute(
                "SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id "
                "WHERE s.token_id=? AND s.user_id=? AND s.expires_at>=?",
                (token_id, int(user_id), int(time.time())),
            ).fetchone()
    except (ValueError, sqlite3.Error) as exc:
        raise HTTPException(status_code=401, detail="Invalid session.") from exc
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid session.")
    if REQUIRE_EMAIL_VERIFICATION and not row["email_verified"]:
        raise HTTPException(status_code=403, detail="Email verification is required.")
    return row


class Credentials(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=120)


class ScanRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    url: str = Field(min_length=1, max_length=4096)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any(character.isspace() for character in parsed.hostname)
        ):
            raise ValueError("Only valid HTTP or HTTPS URLs are accepted.")
        return value


class User(BaseModel):
    name: str
    email: EmailStr


class AuthResponse(BaseModel):
    user: User
    access_token: str
    token_type: str = "bearer"
    verification_required: bool = False


class ScanResponse(BaseModel):
    url: str
    verdict: str
    probability: float
    risk_level: str
    model: str
    explanations: dict[str, bool | int]
    model_scores: dict[str, float] = {}
    decision_threshold: float


class FeedbackRequest(BaseModel):
    scan_url: str = Field(min_length=1, max_length=4096)
    predicted_verdict: str = Field(pattern="^(phishing|legitimate)$")
    analyst_verdict: str = Field(default="uncertain", pattern="^(phishing|legitimate|uncertain)$")
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    comment: str | None = Field(default=None, max_length=1000)
    feedback_text: str | None = Field(default=None, max_length=1000)


def _typed_feedback_verdict(feedback_text: str | None) -> str | None:
    if not feedback_text:
        return None
    text = re.sub(r"[^a-z]+", " ", feedback_text.lower())
    not_phishing = bool(re.search(r"\bnot phishing\b", text))
    phishing = bool(
        re.search(r"\b(phishing|phish|malicious|scam|fraud)\b", text)
    ) and not not_phishing
    legitimate = bool(
        re.search(r"\b(legitimate|legit|safe|benign)\b", text)
    ) or not_phishing
    if phishing and legitimate:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Typed feedback contains conflicting labels. State only legitimate or phishing.",
        )
    if phishing:
        return "phishing"
    if legitimate:
        return "legitimate"
    return None


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


class EmailTokenRequest(BaseModel):
    email: EmailStr


class PasswordResetRequest(BaseModel):
    token: str = Field(min_length=32, max_length=256)
    new_password: str = Field(min_length=12, max_length=128)


def _load_rl_state() -> dict:
    if not RL_STATE_FILE.exists():
        return {
            "thresholds": [0.4, 0.5, 0.6, 0.7, 0.8],
            "q_values": [0.0, 0.0, 0.0, 0.0, 0.2],
        }
    try:
        state = json.loads(RL_STATE_FILE.read_text(encoding="utf-8"))
        if len(state["thresholds"]) != len(state["q_values"]):
            raise ValueError("threshold and q-value lengths differ")
        if not state["thresholds"] or not all(
            0.0 <= float(value) <= 1.0 for value in state["thresholds"]
        ):
            raise ValueError("thresholds must be non-empty probabilities")
        if all(float(value) == 0.0 for value in state["q_values"]):
            midpoint = int(np.argmin(np.abs(np.asarray(state["thresholds"]) - 0.5)))
            state["q_values"][midpoint] = 0.2
        return state
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Adaptive-learning state is invalid.",
        ) from exc


def _save_rl_state(state: dict) -> None:
    RL_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = RL_STATE_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(RL_STATE_FILE)


def _active_threshold(state: dict) -> float:
    return float(state["thresholds"][int(np.argmax(state["q_values"]))])


def _schedule_quantum_adaptation() -> None:
    global ADAPTATION_RUNNING
    with ADAPTATION_LOCK:
        if ADAPTATION_RUNNING:
            return
        with _db() as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE analyst_verdict != 'uncertain'"
            ).fetchone()[0]
            row = connection.execute(
                "SELECT value FROM metrics WHERE key='quantum_feedback_count'"
            ).fetchone()
        trained_count = int(row["value"]) if row else 0
        if count < trained_count + 10:
            return
        ADAPTATION_RUNNING = True

    def worker() -> None:
        global ADAPTATION_RUNNING
        try:
            with _db() as connection:
                rows = connection.execute(
                    "SELECT scan_url, analyst_verdict FROM feedback "
                    "WHERE analyst_verdict != 'uncertain' ORDER BY id"
                ).fetchall()
            adapt_live_model(
                DEFAULT_DATASET,
                QUANTUM_MODEL_FILE,
                [row["scan_url"] for row in rows],
                [
                    int(row["analyst_verdict"] == "phishing") for row in rows
                ],
                quantum_weight=QUANTUM_WEIGHT,
            )
            with _db() as connection:
                connection.execute(
                    "INSERT INTO metrics(key,value,updated_at) VALUES(?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    ("quantum_feedback_count", len(rows), datetime.now(timezone.utc).isoformat()),
                )
            logger.info("Quantum model adapted using %d labeled feedback records", len(rows))
        except Exception:
            logger.exception("Quantum model adaptation failed")
        finally:
            with ADAPTATION_LOCK:
                ADAPTATION_RUNNING = False

    threading.Thread(target=worker, name="quantum-adaptation", daemon=True).start()


def _schedule_classical_retraining(records: list[dict]) -> None:
    global RETRAIN_RUNNING
    if not records:
        return
    with ADAPTATION_LOCK:
        if RETRAIN_RUNNING:
            return
        RETRAIN_RUNNING = True

    def worker() -> None:
        global RETRAIN_RUNNING
        try:
            _retrain_from_feedback(records)
            logger.info("Classical models retrained using %d labeled feedback records", len(records))
        except Exception:
            logger.exception("Classical model retraining failed")
        finally:
            with ADAPTATION_LOCK:
                RETRAIN_RUNNING = False

    threading.Thread(target=worker, name="classical-retraining", daemon=True).start()


def _retrain_from_feedback(records: list[dict]) -> int:
    labeled = [record for record in records if record["analyst_verdict"] != "uncertain"]
    if not labeled:
        return 0

    feature_frame, target, features = build_production_training_data(DEFAULT_DATASET)
    feedback_frame = build_feature_frame(
        pd.DataFrame({"URL": [record["scan_url"] for record in labeled]}),
        "url_raw",
    )
    feedback_target = pd.Series(
        [int(record["analyst_verdict"] == "phishing") for record in labeled],
        dtype=int,
    )
    x_train = pd.concat([feature_frame, feedback_frame[features]], ignore_index=True)
    y_train = pd.concat([target, feedback_target], ignore_index=True)
    models = {
        "A": make_baseline(42),
        "B": make_ensemble(42, trees=100, boosting_iterations=100),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.fit(x_train, y_train)
        temporary = MODEL_DIR / f"model_{name}.joblib.tmp"
        joblib.dump(model, temporary)
        temporary.replace(MODEL_DIR / f"model_{name}.joblib")
    return len(labeled)


def _result_file(name: str) -> Path:
    return ROOT / "results" / name


def _load_json_result(name: str) -> dict:
    path = _result_file(name)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Experiment result is unavailable: {name}",
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Experiment result could not be read: {name}",
        ) from exc


def _optional_json_result(name: str) -> dict | None:
    path = _result_file(name)
    if not path.exists():
        return None
    return _load_json_result(name)


_MODEL_CACHE: dict[str, tuple] = {}


def _models() -> dict[str, object]:
    loaded: dict[str, object] = {}
    for name in ("A", "B"):
        path = MODEL_DIR / f"model_{name}.joblib"
        if not path.exists():
            continue
        stat = path.stat()
        cache_key = (stat.st_mtime_ns, stat.st_size)
        cached = _MODEL_CACHE.get(name)
        if cached is None or cached[0] != cache_key:
            cached = (cache_key, joblib.load(path))
            _MODEL_CACHE[name] = cached
        loaded[name] = cached[1]
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No trained URL models are available.",
        )
    return loaded


_QUANTUM_CACHE: dict[str, object] = {"key": None, "model": None}


def _quantum_model() -> dict:
    global _QUANTUM_CACHE
    if not QUANTUM_MODEL_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantum live inference model is unavailable.",
        )
    try:
        stat = QUANTUM_MODEL_FILE.stat()
        cache_key = (stat.st_mtime_ns, stat.st_size)
        cached = _QUANTUM_CACHE["model"]
        if cached is None or _QUANTUM_CACHE["key"] != cache_key:
            from .quantum_live import load_quantum_model

            cached = load_quantum_model(QUANTUM_MODEL_FILE)
            _QUANTUM_CACHE = {"key": cache_key, "model": cached}
        return cached
    except (OSError, ValueError, ImportError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Quantum live inference model could not be loaded.",
        ) from exc


def _risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "high"
    if probability >= 0.3:
        return "medium"
    return "low"


def _redact_url(value: str) -> str:
    parsed = urlsplit(value)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


@app.get("/api/health")
def health() -> dict[str, bool | str | int | float]:
    classical_available = all(
        (MODEL_DIR / f"model_{name}.joblib").exists() for name in ("A", "B")
    )
    quantum_available = QUANTUM_MODEL_FILE.exists()
    try:
        with _db() as connection:
            scans = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
            feedback_count = connection.execute(
                "SELECT COUNT(*) FROM feedback WHERE analyst_verdict != 'uncertain'"
            ).fetchone()[0]
            db_status = "available"
    except (sqlite3.Error, psycopg.Error):
        scans, feedback_count, db_status = 0, 0, "unavailable"
    return {
        "status": "ok",
        "detection_engine": (
            "available"
            if classical_available and quantum_available
            else "unavailable"
        ),
        "classical_engine": "available" if classical_available else "unavailable",
        "quantum_engine": "available" if quantum_available else "unavailable",
        "database": db_status,
        "scans": scans,
        "feedback_labels": feedback_count,
        "adaptive_threshold": _active_threshold(_load_rl_state()),
    }


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/scans/history")
def scan_history(
    user: sqlite3.Row = Depends(_current_user),
) -> list[dict[str, object]]:
    with _db() as connection:
        rows = connection.execute(
            "SELECT scan_url,verdict,probability,decision_threshold,created_at "
            "FROM scans WHERE user_id=? ORDER BY id DESC LIMIT 50",
            (user["id"],),
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/metrics")
def metrics(user: sqlite3.Row = Depends(_current_user)) -> dict[str, object]:
    with _db() as connection:
        totals = connection.execute(
            "SELECT COUNT(*) total, COALESCE(AVG(probability),0) average_probability,"
            "SUM(CASE WHEN verdict='phishing' THEN 1 ELSE 0 END) phishing "
            "FROM scans WHERE user_id=?",
            (user["id"],),
        ).fetchone()
        feedback_totals = connection.execute(
            "SELECT COUNT(*) total, COALESCE(AVG(reward),0) average_reward "
            "FROM feedback WHERE user_id=?",
            (user["id"],),
        ).fetchone()
    return {
        "scans": dict(totals),
        "feedback": dict(feedback_totals),
        "quantum_weight": QUANTUM_WEIGHT,
        "adaptive_threshold": _active_threshold(_load_rl_state()),
    }


@app.post("/api/auth/login", response_model=AuthResponse)
def login(credentials: Credentials) -> AuthResponse:
    _init_db()
    with _db() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email=?", (str(credentials.email).lower(),)
        ).fetchone()
    if row is None or not _password_matches(credentials.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if REQUIRE_EMAIL_VERIFICATION and not row["email_verified"]:
        raise HTTPException(status_code=403, detail="Verify your email before signing in.")
    return AuthResponse(
        user=User(name=row["name"], email=row["email"]),
        access_token=_token(row["id"]),
    )


@app.post("/api/auth/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(credentials: Credentials) -> AuthResponse:
    _init_db()
    email = str(credentials.email).lower()
    name = credentials.name or email.split("@")[0]
    try:
        with _db() as connection:
            cursor = connection.execute(
                "INSERT INTO users(email,name,password_hash,created_at) VALUES(?,?,?,?) "
                "RETURNING id",
                (email, name, _password_hash(credentials.password), datetime.now(timezone.utc).isoformat()),
            )
            user_id = cursor.fetchone()[0]
    except (sqlite3.IntegrityError, psycopg.errors.UniqueViolation) as exc:
        raise HTTPException(status_code=409, detail="An account already exists for this email.") from exc
    verification_required = REQUIRE_EMAIL_VERIFICATION
    if verification_required:
        token = _email_token(int(user_id), "verify")
        try:
            _send_email(
                email,
                "Verify your QPhish account",
                f"Verify your account: {PUBLIC_API_URL}/api/auth/verify-email?token={token}",
            )
        except (RuntimeError, smtplib.SMTPException, OSError) as exc:
            with _db() as connection:
                connection.execute("DELETE FROM email_tokens WHERE user_id=?", (user_id,))
                connection.execute("DELETE FROM users WHERE id=?", (user_id,))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return AuthResponse(
        user=User(name=name, email=email),
        access_token="" if verification_required else _token(int(user_id)),
        verification_required=verification_required,
    )


@app.get("/api/auth/me", response_model=User)
def me(user: sqlite3.Row = Depends(_current_user)) -> User:
    return User(name=user["name"], email=user["email"])


@app.post("/api/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    authorization: str | None = Header(default=None),
    user: sqlite3.Row = Depends(_current_user),
) -> Response:
    del user
    _revoke_token(authorization[7:])
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/api/auth/password")
def change_password(
    request: PasswordChangeRequest,
    user: sqlite3.Row = Depends(_current_user),
) -> dict[str, bool]:
    if not _password_matches(request.current_password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Current password is incorrect.")
    if request.current_password == request.new_password:
        raise HTTPException(status_code=400, detail="New password must be different.")
    with _db() as connection:
        connection.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (_password_hash(request.new_password), user["id"]),
        )
        connection.execute("DELETE FROM sessions WHERE user_id=?", (user["id"],))
    return {"updated": True}


@app.get("/api/auth/verify-email")
def verify_email(token: str = Query(min_length=32, max_length=256)) -> dict[str, bool]:
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    with _db() as connection:
        row = connection.execute(
            "SELECT user_id FROM email_tokens WHERE token_hash=? AND purpose='verify' "
            "AND used_at IS NULL AND expires_at>=?",
            (token_hash, int(time.time())),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
        connection.execute("UPDATE users SET email_verified=1 WHERE id=?", (row["user_id"],))
        connection.execute(
            "UPDATE email_tokens SET used_at=? WHERE token_hash=?",
            (datetime.now(timezone.utc).isoformat(), token_hash),
        )
    return {"verified": True}


@app.post("/api/auth/password-reset/request")
def request_password_reset(request: EmailTokenRequest) -> dict[str, bool]:
    with _db() as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE email=?", (str(request.email).lower(),)
        ).fetchone()
    if row is not None:
        token = _email_token(row["id"], "reset")
        try:
            _send_email(
                str(request.email).lower(),
                "Reset your QPhish password",
                f"Reset your password: {PUBLIC_API_URL}/reset-password?token={token}",
            )
        except (RuntimeError, smtplib.SMTPException, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"accepted": True}


@app.post("/api/auth/password-reset")
def reset_password(request: PasswordResetRequest) -> dict[str, bool]:
    token_hash = hashlib.sha256(request.token.encode()).hexdigest()
    with _db() as connection:
        row = connection.execute(
            "SELECT user_id FROM email_tokens WHERE token_hash=? AND purpose='reset' "
            "AND used_at IS NULL AND expires_at>=?",
            (token_hash, int(time.time())),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token.")
        connection.execute(
            "UPDATE users SET password_hash=? WHERE id=?",
            (_password_hash(request.new_password), row["user_id"]),
        )
        connection.execute("DELETE FROM sessions WHERE user_id=?", (row["user_id"],))
        connection.execute(
            "UPDATE email_tokens SET used_at=? WHERE token_hash=?",
            (datetime.now(timezone.utc).isoformat(), token_hash),
        )
    return {"reset": True}


@app.get("/api/models/comparison")
def model_comparison() -> dict:
    first = _load_json_result("first_experiment.json")
    quantum = _load_json_result("quantum_ensemble_experiment.json")
    metrics = first.get("metrics", {})
    response = {
        "models": [
            {
                "id": "A",
                "name": "Logistic regression",
                "type": "classical",
                "scope": "Full domain-held-out test set",
                "metrics": metrics.get("A", {}),
            },
            {
                "id": "B",
                "name": "Soft-voting ensemble",
                "type": "classical",
                "scope": "Full domain-held-out test set",
                "metrics": metrics.get("B", {}),
            },
            {
                "id": "quantum-ensemble",
                "name": "Classical + quantum ensemble",
                "type": "hybrid",
                "scope": "Held-out balanced subset",
                "metrics": quantum.get("metrics", {}),
                "component_metrics": quantum.get("component_metrics", {}),
                "weights": quantum.get("weights", {}),
                "circuit": quantum.get("circuit", {}),
                "evaluation_only": False,
            },
        ],
    }
    matrix = _optional_json_result("quantum_rl_configuration_experiment.json")
    if matrix is not None:
        response["configuration_matrix"] = {
            "experiment": matrix.get("experiment"),
            "circuit": matrix.get("circuit", {}),
            "drift": matrix.get("drift"),
            "configurations": matrix.get("configurations", []),
        }
    response["final_risk_score"] = {
        "live_model": "75% classical + 25% quantum interference"
        if QUANTUM_MODEL_FILE.exists()
        else "A+B ensemble",
        "quantum_included": QUANTUM_MODEL_FILE.exists(),
        "reason": (
            "Live URL scoring blends the persisted VQC probability with the "
            "classical ensemble."
            if QUANTUM_MODEL_FILE.exists()
            else "Quantum live inference model is unavailable."
        ),
    }
    return response


@app.post("/api/feedback", status_code=status.HTTP_201_CREATED)
def feedback(
    request: FeedbackRequest,
    user: sqlite3.Row = Depends(_current_user),
) -> dict[str, str | bool | int | float | None]:
    parsed = urlsplit(request.scan_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only valid HTTP or HTTPS URLs are accepted.",
        )
    typed_verdict = _typed_feedback_verdict(request.feedback_text)
    if (
        typed_verdict is not None
        and request.analyst_verdict != "uncertain"
        and typed_verdict != request.analyst_verdict
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Typed feedback conflicts with the selected analyst verdict.",
        )
    analyst_verdict = typed_verdict or request.analyst_verdict
    stored_comment = request.feedback_text or request.comment
    with MODEL_LOCK:
        state = _load_rl_state()
        active_threshold = _active_threshold(state)
        predicted = request.predicted_verdict == "phishing"
        actual = (
            None
            if analyst_verdict == "uncertain"
            else analyst_verdict == "phishing"
        )
        reward = 0.0 if actual is None else (1.0 if predicted == actual else -1.0)
        observed_threshold = (
            active_threshold
            if request.decision_threshold is None
            else request.decision_threshold
        )
        action = int(
            np.argmin(np.abs(np.asarray(state["thresholds"]) - observed_threshold))
        )
        if actual is not None:
            state["q_values"][action] += 0.2 * (reward - state["q_values"][action])
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scan_url": request.scan_url,
            "predicted_verdict": request.predicted_verdict,
            "analyst_verdict": analyst_verdict,
            "probability": request.probability,
            "decision_threshold": observed_threshold,
            "comment": stored_comment,
        }
        with _db() as connection:
            connection.execute(
                "INSERT INTO feedback(user_id,scan_url,predicted_verdict,analyst_verdict,"
                "probability,decision_threshold,comment,reward,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    user["id"], request.scan_url, request.predicted_verdict,
                    analyst_verdict, request.probability, observed_threshold,
                    stored_comment, reward, record["timestamp"],
                ),
            )
            stored_feedback = connection.execute(
                "SELECT scan_url, analyst_verdict FROM feedback ORDER BY id"
            ).fetchall()
        _save_rl_state(state)
        labeled_feedback = [dict(item) for item in stored_feedback]
        trained_labels = sum(
            item["analyst_verdict"] != "uncertain" for item in labeled_feedback
        )
        _schedule_classical_retraining(labeled_feedback)
        _schedule_quantum_adaptation()
    return {
        "accepted": True,
        "message": (
            "Feedback accepted and models retrained."
            if trained_labels
            else "Feedback accepted; uncertain labels are not used for training."
        ),
        "training_signal": analyst_verdict,
        "trained_feedback_labels": trained_labels,
        "adaptive_threshold": _active_threshold(state),
        "reward": reward,
    }


@app.post("/api/scans", response_model=ScanResponse)
def scan(
    request: ScanRequest,
    user: sqlite3.Row = Depends(_current_user),
) -> ScanResponse:
    features = extract_url_features(request.url)
    models = _models()
    quantum_model = _quantum_model()
    # The training pipeline defines target 1 as phishing.
    model_scores = {
        name: float(model.predict_proba(features)[:, 1][0])
        for name, model in models.items()
    }
    weight_total = sum(CLASSICAL_WEIGHTS.get(name, 1.0) for name in model_scores)
    classical_probability = sum(
        CLASSICAL_WEIGHTS.get(name, 1.0) * score
        for name, score in model_scores.items()
    ) / weight_total
    quantum_probability = predict_quantum_probability(quantum_model, features)
    probability = (
        (1 - QUANTUM_WEIGHT) * classical_probability
        + QUANTUM_WEIGHT * quantum_probability
    )
    trusted_domain_calibration = (
        is_trusted_domain_url(request.url)
        and not bool(features["IsDomainIP"].iloc[0])
        and not bool(features["HasObfuscation"].iloc[0])
        # Fragments are never sent to the server, so only pre-fragment
        # special characters disqualify a trusted host from calibration.
        and sum(
            character in OTHER_SPECIAL_CHARACTERS
            for character in request.url.split("#", 1)[0]
        )
        == 0
        and int(features["NoOfSubDomain"].iloc[0]) <= 2
    )
    if trusted_domain_calibration:
        probability = min(probability, 0.25)
    # Phishing is the positive class (target 1) in all trained models.
    decision_threshold = _active_threshold(_load_rl_state())
    is_phishing = probability >= decision_threshold
    response = ScanResponse(
        url=_redact_url(request.url),
        verdict="phishing" if is_phishing else "legitimate",
        probability=probability,
        risk_level=_risk_level(probability),
        model="classical-quantum ensemble",
        explanations={
            "uses_https": bool(features["IsHTTPS"].iloc[0]),
            "is_domain_ip": bool(features["IsDomainIP"].iloc[0]),
            "has_obfuscation": bool(features["HasObfuscation"].iloc[0]),
            "url_length": int(features["URLLength"].iloc[0]),
            "subdomain_count": int(features["NoOfSubDomain"].iloc[0]),
            "special_character_count": int(features["NoOfOtherSpecialCharsInURL"].iloc[0]),
            "trusted_domain_calibration": trusted_domain_calibration,
        },
        model_scores=model_scores
        | {
            "quantum": quantum_probability,
            "hybrid": probability,
        },
        decision_threshold=decision_threshold,
    )
    with _db() as connection:
        connection.execute(
            "INSERT INTO scans(user_id,scan_url,verdict,probability,decision_threshold,model_scores,created_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (
                user["id"], request.url, response.verdict, response.probability,
                response.decision_threshold, json.dumps(response.model_scores),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    SCAN_COUNT.labels(response.verdict).inc()
    return response
