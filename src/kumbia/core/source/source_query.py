"""
   Source-neutral descriptor of "what to fetch". address is whatever locates
   the data for a given source: a URL for JSON/Elasticsearch, a sqlite path for
   SQL. New fields can be added here without changing DataSource.fetch's
   signature or any call site.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


@dataclass(frozen=True)
class SourceQuery:
    address: str
    statement: str | None = None       # query text for sources that need it
    credentials: Credentials | None = None