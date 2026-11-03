# QPhish deployment

1. Copy `.env.example` to `.env`.
2. Set a random `QPHISH_AUTH_SECRET` with at least 32 characters.
3. Set `QPHISH_ALLOWED_HOSTS` and `QPHISH_FRONTEND_ORIGINS` to the real domains.
4. Put the TLS certificate and key at `deploy/certs/fullchain.pem` and
   `deploy/certs/privkey.pem`.
5. Build the frontend with `npm --prefix frontend run build`.
6. Start the stack with `docker compose up -d --build`.
7. Run the external dependency preflight:

   ```powershell
   python deploy/preflight.py
   ```

For a real host, obtain certificates before starting the gateway, for example
with Certbot:

```bash
sudo certbot certonly --standalone -d api.example.com -d app.example.com
cp /etc/letsencrypt/live/app.example.com/fullchain.pem deploy/certs/fullchain.pem
cp /etc/letsencrypt/live/app.example.com/privkey.pem deploy/certs/privkey.pem
```

Then run:

```bash
docker compose up -d --build
docker compose ps
curl -f https://api.example.com/api/health
```

   This checks the configured PostgreSQL connection, SMTP authentication, TLS
   certificate files, production secrets/domains, and Docker Compose.

The deployment uses PostgreSQL from `DATABASE_URL` and mounts `models/` so
asynchronous model adaptation survives container restarts. SQLite remains the
local fallback when `DATABASE_URL` is unset. PostgreSQL is required for
multiple API replicas.

Set the SMTP variables and `QPHISH_REQUIRE_EMAIL_VERIFICATION=true` to enable
verification and password-reset emails. The API never reports whether a reset
email address exists.

Prometheus scrapes `/metrics` internally. The public gateway restricts that
endpoint to private network ranges.

The preflight intentionally does not send an email or print credentials. It
only authenticates to SMTP and runs `SELECT 1` against PostgreSQL.
