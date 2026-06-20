"""Praxos LMS backend — FastAPI + SQLAlchemy + Alembic.

A self-contained service (Python 3.9-compatible) that mirrors the web app's
domain. Defaults to SQLite so it runs and tests without external services;
point DATABASE_URL at Supabase Postgres for production.
"""
