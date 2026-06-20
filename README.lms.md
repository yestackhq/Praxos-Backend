# Praxos LMS backend

FastAPI + SQLAlchemy + Alembic service that mirrors the web app's domain
(`web/src/lib/mock.ts`). Self-contained and Python 3.9-compatible. Defaults to
SQLite so it runs and tests with **no external services**; point `DATABASE_URL`
at Supabase Postgres for production.

> The `backend/backend/` tree is the upstream benavlabs/FastAPI-boilerplate
> (Redis + Postgres + taskiq, Python 3.11+). `lms_app/` is the focused,
> runnable LMS service used by the app.

## Run

```bash
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                       # optional; SQLite works without it
.venv/bin/alembic upgrade head             # or rely on create_all on startup
.venv/bin/uvicorn lms_app.main:app --reload --port 8000
```

Open http://localhost:8000/docs.

## Test

```bash
.venv/bin/python -m pytest tests_lms -q
```

## Endpoints (all under `/api`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service + auth status |
| GET | `/workspace` | Current workspace |
| GET | `/learner/home` | Stats + learning path |
| GET | `/learner/sessions` | Past sessions |
| GET | `/learner/documents` | Assigned documents |
| GET | `/admin/overview` | KPIs, cohort health, at-risk people |
| GET | `/admin/people` | All people |
| GET | `/admin/teams` | Teams |
| GET | `/admin/cohorts` | Cohorts |
| GET | `/documents` | All documents |
| GET | `/documents/{id}/plan` | Teaching plan (modules) |

## Auth

When `CLERK_JWKS_URL` is set, every `/api/learner|admin|documents` route requires a
valid Clerk session JWT (`Authorization: Bearer <token>`), verified against the
Clerk JWKS (`lms_app/auth.py`). Unset → open (review/dev mode).

## Original-PDF storage (client-side, no server secret key)

The browser uploads the original PDF straight to Supabase Storage with the
**publishable** key (`web/src/lib/docStorage.ts`) — the backend never needs a
service/secret key. The backend still receives the bytes once to extract + embed
the text; it only records the returned `storage_path`.

One-time Supabase setup (SQL editor): create a `documents` bucket and, because we
auth with Clerk (no Supabase Auth session → uploads run as the `anon` role),
allow anon insert/read:

```sql
insert into storage.buckets (id, name, public)
values ('documents', 'documents', true)
on conflict (id) do nothing;

create policy "anon upload documents"
  on storage.objects for insert to anon
  with check (bucket_id = 'documents');

create policy "anon read documents"
  on storage.objects for select to anon
  using (bucket_id = 'documents');
```

Frontend env (`web/.env`): `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`,
and optionally `VITE_SUPABASE_BUCKET` (defaults to `documents`).
