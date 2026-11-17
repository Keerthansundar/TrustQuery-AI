"""Database execution safety layer: connection pooling, read-only enforcement, and plan cost checks."""

import logging
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from src.config import settings

logger = logging.getLogger(__name__)


class QueryTooExpensiveError(ValueError):
    """Raised when an EXPLAIN plan cost exceeds MAX_PLAN_COST."""

    pass


def get_conninfo(role: str = "analyst_ro") -> str:
    """Builds PostgreSQL connection string for the requested role."""
    host = settings.POSTGRES_HOST
    port = settings.POSTGRES_PORT
    dbname = settings.POSTGRES_DB

    if role == "analyst_pii_ro":
        user = "analyst_pii_ro"
        password = settings.ANALYST_PII_RO_PASSWORD
    elif role == "app_rw":
        user = "app_rw"
        password = settings.APP_RW_PASSWORD
    elif role == "owner":
        user = settings.POSTGRES_USER
        password = settings.POSTGRES_PASSWORD
    else:
        user = "analyst_ro"
        password = settings.ANALYST_RO_PASSWORD

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


class DatabaseManager:
    """Manages connection pools and safe query execution."""

    def __init__(self) -> None:
        self._analyst_pool: ConnectionPool | None = None
        self._analyst_pii_pool: ConnectionPool | None = None
        self._app_rw_pool: ConnectionPool | None = None

    def _get_pool(self, role: str) -> ConnectionPool:
        if role == "analyst_pii_ro":
            if self._analyst_pii_pool is None:
                self._analyst_pii_pool = ConnectionPool(
                    get_conninfo("analyst_pii_ro"), min_size=1, max_size=10, timeout=5.0, open=True
                )
            return self._analyst_pii_pool
        elif role == "app_rw":
            if self._app_rw_pool is None:
                self._app_rw_pool = ConnectionPool(
                    get_conninfo("app_rw"), min_size=1, max_size=10, timeout=5.0, open=True
                )
            return self._app_rw_pool
        else:
            if self._analyst_pool is None:
                self._analyst_pool = ConnectionPool(
                    get_conninfo("analyst_ro"), min_size=1, max_size=10, timeout=5.0, open=True
                )
            return self._analyst_pool

    def check_health(self) -> bool:
        """Returns True if PostgreSQL is reachable and healthy."""
        try:
            with psycopg.connect(get_conninfo("analyst_ro"), connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    return cur.fetchone()[0] == 1
        except Exception:
            return False

    def run_query(
        self,
        sql: str,
        allow_pii: bool | None = None,
        max_rows: int | None = None,
        max_cost: float | None = None,
    ) -> list[dict[str, Any]]:
        """Safely executes SQL query within a read-only transaction.

        Enforces:
        - Connection role (analyst_ro vs analyst_pii_ro)
        - Statement timeout (5s) and lock timeout (2s)
        - EXPLAIN plan cost validation (rejects cartesian explosions)
        - Row truncation (fetchmany capped at MAX_ROWS)
        """
        pii_enabled = allow_pii if allow_pii is not None else settings.ALLOW_PII
        role = "analyst_pii_ro" if pii_enabled else "analyst_ro"
        pool = self._get_pool(role)

        cost_threshold = max_cost if max_cost is not None else settings.MAX_PLAN_COST
        limit_rows = max_rows or settings.MAX_ROWS

        clean_sql = sql.strip().rstrip(";")

        with pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # Enforce strict session safety parameters
                    cur.execute("SET LOCAL default_transaction_read_only = on;")
                    cur.execute("SET LOCAL statement_timeout = '5s';")
                    cur.execute("SET LOCAL lock_timeout = '2s';")

                    # 1. Pre-flight EXPLAIN (FORMAT JSON) check for plan cost safety
                    cur.execute(f"EXPLAIN (FORMAT JSON) {clean_sql}")
                    explain_data = cur.fetchone()[0]
                    total_cost = 0.0
                    if isinstance(explain_data, list) and explain_data:
                        plan_dict = explain_data[0].get("Plan", {})
                        total_cost = float(plan_dict.get("Total Cost", 0.0))

                    if total_cost > cost_threshold:
                        raise QueryTooExpensiveError(
                            f"Query plan cost ({total_cost:.1f}) exceeds maximum allowed cost "
                            f"({cost_threshold:.1f}). Query is too expensive; please add filters or aggregations."
                        )

                    # 2. Execute actual query
                    cur.execute(clean_sql)

                    if not cur.description:
                        return []

                    colnames = [desc[0] for desc in cur.description]
                    rows = cur.fetchmany(limit_rows)
                    return [dict(zip(colnames, row, strict=False)) for row in rows]

    def close(self) -> None:
        """Closes all connection pools."""
        for pool in (self._analyst_pool, self._analyst_pii_pool, self._app_rw_pool):
            if pool is not None:
                try:
                    pool.close()
                except Exception:
                    pass


# Global database manager instance
db_manager = DatabaseManager()
