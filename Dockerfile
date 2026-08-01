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
# Bind :: (dual-stack), NOT 0.0.0.0. Railway's private network is IPv6-only, so
# an IPv4-only listener is unreachable at <service>.railway.internal — the
# service would only work through its public URL, which is exactly what we are
# removing. On Linux :: accepts IPv4-mapped connections too, so the public edge
# keeps working unchanged.
CMD ["sh", "-c", "uvicorn lms_app.main:app --host :: --port ${PORT:-8000}"]
