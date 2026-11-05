# Deployment

The production stack lives in `docker-compose.yml` with configs in `deploy/`:

| Service | Role |
| --- | --- |
| `api` | FastAPI (uvicorn) with mounted `models/` and `data/` volumes |
| `frontend` | nginx serving the built Vite bundle |
| `postgres` | PostgreSQL 16 (used when `DATABASE_URL` is set) |
| `prometheus` | internal metrics scraping of `/metrics` |
| `gateway` | nginx TLS entry point, rate limiting, `/metrics` restricted to private networks |

## Steps

1. Copy `.env.example` to `.env`.
2. Set a random `QPHISH_AUTH_SECRET` (≥32 characters).
3. Set `QPHISH_ALLOWED_HOSTS` and `QPHISH_FRONTEND_ORIGINS` to your real
   domains.
4. Place TLS certificates at `deploy/certs/fullchain.pem` and
   `deploy/certs/privkey.pem` (e.g. via `certbot certonly --standalone`).
5. Build the frontend: `npm --prefix frontend run build`.
6. Start the stack: `docker compose up -d --build`.
7. Verify external dependencies:

   ```bash
   python deploy/preflight.py
   ```

   The preflight checks the PostgreSQL connection (`SELECT 1` only), SMTP
   authentication, TLS files, production secrets/domains, and Docker Compose.
   It never sends an email or prints credentials.

8. Smoke test: `curl -f https://api.example.com/api/health`.

## Notes

- SQLite remains the fallback when `DATABASE_URL` is unset; PostgreSQL is
  required for multiple API replicas.
- `models/` is mounted so feedback-driven retraining survives restarts.
- Set the SMTP variables and `QPHISH_REQUIRE_EMAIL_VERIFICATION=true` to
  enable verification and password-reset emails.
- Retraining is synchronous in the feedback path; move it to a job queue
  before high-volume production use.
- Prometheus scrapes `/metrics` internally; the gateway blocks it from the
  public internet.
