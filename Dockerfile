# Praxos LMS backend — runs the self-contained FastAPI service in lms_app/.
# (The benavlabs boilerplate under backend/ is unused; we run lms_app directly.)
FROM python:3.11-slim

WORKDIR /app

# build-essential covers any wheels that need compiling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# psycopg[binary] is commented out in requirements.txt (local dev uses SQLite);
# the Supabase Postgres deploy needs the driver, so install it explicitly.
RUN pip install --no-cache-dir -r requirements.txt "psycopg[binary]>=3.1"

COPY . .

ENV PORT=8000
# serve.py binds ONE dual-stack socket. Neither uvicorn --host value works here:
# 0.0.0.0 is unreachable over Railway's IPv6-only private network, and :: comes
# up IPv6-only in this container so the platform's IPv4 healthcheck never lands.
# See serve.py for the evidence.
CMD ["python", "serve.py"]
