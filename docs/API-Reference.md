# QPhish API Reference

Base URL: `http://localhost:8000` (development) or the configured API host.
All endpoints except health, register, and login require
`Authorization: Bearer <access_token>`.

## Auth

| Method | Path | Description |
| --- | --- | --- |
| POST | `/api/auth/register` | Create an account. Body: `{"email", "password", "name"}`. Returns 201; may require email verification. |
| POST | `/api/auth/login` | Exchange credentials for a bearer token. Body: `{"email", "password"}`. Returns `{"access_token", "user"}`. |
| POST | `/api/auth/logout` | Revoke the current token. 204. |
| GET | `/api/auth/me` | Current user profile. |
| GET | `/api/auth/verify-email` | Confirm an email-verification token (query param). |
| POST | `/api/auth/password-reset/request` | Send a reset email. Always returns the same response, whether or not the address exists. |
| POST | `/api/auth/password-reset` | Complete a reset with token + new password. |
| POST | `/api/auth/password` | Change the password (authenticated). |

## Scanning

### POST `/api/scans`

Body: `{"url": "https://example.com"}` — only HTTP(S) URLs accepted.

```json
{
  "url": "https://www.google.com/search",
  "verdict": "legitimate",
  "probability": 0.1508,
  "risk_level": "low",
  "model": "classical-quantum ensemble",
  "decision_threshold": 0.4,
  "explanations": {
    "uses_https": true, "is_domain_ip": false, "has_obfuscation": false,
    "url_length": 55, "subdomain_count": 1, "special_character_count": 0,
    "trusted_domain_calibration": true
  },
  "model_scores": { "A": 0.0007, "B": 0.0003, "quantum": 0.6016, "hybrid": 0.1508 }
}
```

Displayed URLs are redacted (query strings and fragments removed).

### GET `/api/scans/history`

The authenticated user's scans, newest first. Each row: `scan_url`, `verdict`,
`probability`, `created_at`.

## Feedback

### POST `/api/feedback`

```json
{
  "scan_url": "https://example.com",
  "predicted_verdict": "phishing",
  "analyst_verdict": "legitimate",
  "probability": 0.82,
  "decision_threshold": 0.4,
  "comment": "checked the WHOIS"
}
```

`analyst_verdict` ∈ `legitimate | phishing | uncertain`; alternatively send
`feedback_text` and the server extracts an explicit label. Returns the stored
record, the trained-label count, the new adaptive threshold, and the reward.

## Models and metrics

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/models/comparison` | Saved experiment metrics for models A, B and the hybrid ensemble, the live fusion description, and (when present) the quantum × RL configuration matrix. |
| GET | `/api/metrics` | Per-user counters: scans total/phishing, average probability, feedback total/reward, adaptive threshold, quantum weight. |
| GET | `/api/health` | Public health probe: engine availability, database, global counters. |
| GET | `/metrics` | Prometheus exposition (internal traffic only in deployment). |

## Errors

- `401` invalid or expired token; `403` unverified account; `404` unknown
  record; `422` validation failure (FastAPI detail describes the field);
  `429` rate limit exceeded at the gateway.
- Error bodies use `{"detail": "message"}`.
