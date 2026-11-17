"""Database verification script.

Asserts exact counts, expected profiles, foreign key integrity,
and security controls (role grants, column-level security, read-only status, timeouts).
"""

import os
import sys
from datetime import date
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Load .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DBNAME = os.getenv("POSTGRES_DB", "text2sql")
OWNER_USER = os.getenv("POSTGRES_USER", "owner")
OWNER_PASS = os.getenv("POSTGRES_PASSWORD", "change_this_owner_password")
ANALYST_RO_PASS = os.getenv("ANALYST_RO_PASSWORD", "change_this_analyst_ro_password")


def get_owner_conn():
    return psycopg.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user=OWNER_USER,
        password=OWNER_PASS,
        autocommit=True,
    )


def get_analyst_conn():
    return psycopg.connect(
        host=HOST,
        port=PORT,
        dbname=DBNAME,
        user="analyst_ro",
        password=ANALYST_RO_PASS,
        autocommit=True,
    )


def verify_row_counts(conn):
    print("Verifying exact row counts...")
    expected_counts = {
        "categories": 15,
        "customers": 1000,
        "products": 120,
        "orders": 5000,
        "order_items": 11953,
        "payments": 5000,
        "shipments": 4746,
    }
    with conn.cursor() as cur:
        for table, expected in expected_counts.items():
            cur.execute(f"SELECT COUNT(*) FROM public.{table};")
            actual = cur.fetchone()[0]
            assert actual == expected, (
                f"Row count mismatch for {table}: expected {expected}, got {actual}"
            )
            print(f"  [OK] public.{table}: {actual} rows")


def verify_profile(conn):
    print("Verifying dataset profile and business integrity...")
    with conn.cursor() as cur:
        # Date range
        cur.execute("SELECT MIN(order_date), MAX(order_date) FROM public.orders;")
        min_date, max_date = cur.fetchone()
        assert min_date == date(2024, 11, 1), f"Unexpected min order_date: {min_date}"
        assert max_date == date(2026, 10, 29), f"Unexpected max order_date: {max_date}"
        print(f"  [OK] order_date range: {min_date} to {max_date}")

        # Status counts
        cur.execute("SELECT status, COUNT(*) FROM public.orders GROUP BY status;")
        status_counts = dict(cur.fetchall())
        expected_statuses = {
            "delivered": 4446,
            "returned": 253,
            "cancelled": 253,
            "shipped": 47,
            "placed": 1,
        }
        for st, exp in expected_statuses.items():
            actual = status_counts.get(st, 0)
            assert actual == exp, (
                f"Order status count mismatch for '{st}': expected {exp}, got {actual}"
            )
        print("  [OK] Order status distribution matches expected counts exactly")

        # 139 customers with zero orders
        cur.execute(
            """
            SELECT COUNT(*)
            FROM public.customers c
            LEFT JOIN public.orders o ON c.customer_id = o.customer_id
            WHERE o.order_id IS NULL;
            """
        )
        zero_order_custs = cur.fetchone()[0]
        assert zero_order_custs == 139, (
            f"Expected 139 customers with 0 orders, found {zero_order_custs}"
        )
        print(f"  [OK] Customers with zero orders: {zero_order_custs}")

        # 6 products never sold
        cur.execute(
            """
            SELECT COUNT(*)
            FROM public.products p
            LEFT JOIN public.order_items oi ON p.product_id = oi.product_id
            WHERE oi.order_item_id IS NULL;
            """
        )
        unsold_prods = cur.fetchone()[0]
        assert unsold_prods == 6, f"Expected 6 products never sold, found {unsold_prods}"
        print(f"  [OK] Products never sold: {unsold_prods}")

        # 103 shipments with NULL delivered_date
        cur.execute("SELECT COUNT(*) FROM public.shipments WHERE delivered_date IS NULL;")
        null_delivered = cur.fetchone()[0]
        assert null_delivered == 103, (
            f"Expected 103 shipments with NULL delivered_date, found {null_delivered}"
        )
        print(f"  [OK] Shipments with NULL delivered_date: {null_delivered}")

        # Foreign Key integrity checks
        fk_checks = [
            (
                "categories.parent_category_id -> categories.category_id",
                "SELECT COUNT(*) FROM public.categories WHERE parent_category_id IS NOT NULL AND parent_category_id NOT IN (SELECT category_id FROM public.categories)",
            ),
            (
                "products.category_id -> categories.category_id",
                "SELECT COUNT(*) FROM public.products WHERE category_id NOT IN (SELECT category_id FROM public.categories)",
            ),
            (
                "orders.customer_id -> customers.customer_id",
                "SELECT COUNT(*) FROM public.orders WHERE customer_id NOT IN (SELECT customer_id FROM public.customers)",
            ),
            (
                "order_items.order_id -> orders.order_id",
                "SELECT COUNT(*) FROM public.order_items WHERE order_id NOT IN (SELECT order_id FROM public.orders)",
            ),
            (
                "order_items.product_id -> products.product_id",
                "SELECT COUNT(*) FROM public.order_items WHERE product_id NOT IN (SELECT product_id FROM public.products)",
            ),
            (
                "payments.order_id -> orders.order_id",
                "SELECT COUNT(*) FROM public.payments WHERE order_id NOT IN (SELECT order_id FROM public.orders)",
            ),
            (
                "shipments.order_id -> orders.order_id",
                "SELECT COUNT(*) FROM public.shipments WHERE order_id NOT IN (SELECT order_id FROM public.orders)",
            ),
        ]
        for name, query in fk_checks:
            cur.execute(query)
            orphans = cur.fetchone()[0]
            assert orphans == 0, f"FK violation detected for {name}: {orphans} orphaned records"
        print("  [OK] All Foreign Key constraints verified with zero orphans")

        # Payment consistency: amount = items total + shipping_fee within 0.01
        cur.execute(
            """
            SELECT COUNT(*)
            FROM public.payments p
            JOIN (
                SELECT order_id, SUM(quantity * unit_price * (1 - discount_pct / 100.0)) AS items_total
                FROM public.order_items
                GROUP BY order_id
            ) oi ON p.order_id = oi.order_id
            JOIN public.orders o ON p.order_id = o.order_id
            WHERE ABS(p.amount - (oi.items_total + o.shipping_fee)) > 0.01;
            """
        )
        inconsistent_payments = cur.fetchone()[0]
        assert inconsistent_payments == 0, (
            f"Found {inconsistent_payments} payments inconsistent with item total + shipping fee"
        )
        print("  [OK] Payment amounts match item totals + shipping fee within 0.01")


def verify_security():
    print("Verifying database security controls for analyst_ro role...")
    with get_analyst_conn() as conn:
        with conn.cursor() as cur:
            # 1. Non-PII SELECT works
            cur.execute(
                "SELECT customer_id, first_name, last_name, city FROM public.customers LIMIT 1;"
            )
            row = cur.fetchone()
            assert row is not None, "Failed to read non-PII columns from customers"
            print("  [OK] Non-PII columns accessible")

            # 2. PII access to email blocked
            try:
                cur.execute("SELECT email FROM public.customers LIMIT 1;")
                raise AssertionError(
                    "Security breach: analyst_ro was able to SELECT customers.email!"
                )
            except psycopg.errors.InsufficientPrivilege:
                print("  [OK] Access to customers.email successfully blocked")

            # 3. PII access to phone blocked
            try:
                cur.execute("SELECT phone FROM public.customers LIMIT 1;")
                raise AssertionError(
                    "Security breach: analyst_ro was able to SELECT customers.phone!"
                )
            except psycopg.errors.InsufficientPrivilege:
                print("  [OK] Access to customers.phone successfully blocked")

            # 4. Write attempts blocked
            try:
                cur.execute(
                    "INSERT INTO public.categories (category_id, category_name) VALUES (99999, 'Test');"
                )
                raise AssertionError(
                    "Security breach: analyst_ro was able to INSERT into categories!"
                )
            except (psycopg.errors.InsufficientPrivilege, psycopg.errors.ReadOnlySqlTransaction):
                print("  [OK] Write attempts successfully blocked by read-only permissions")

            # 5. Access to schema rag blocked
            try:
                cur.execute("SELECT * FROM rag.documents LIMIT 1;")
                raise AssertionError(
                    "Security breach: analyst_ro was able to access rag.documents!"
                )
            except psycopg.errors.InsufficientPrivilege:
                print("  [OK] Access to rag.documents blocked")

            # 6. Access to schema app blocked
            try:
                cur.execute("SELECT * FROM app.sessions LIMIT 1;")
                raise AssertionError("Security breach: analyst_ro was able to access app.sessions!")
            except psycopg.errors.InsufficientPrivilege:
                print("  [OK] Access to app.sessions blocked")

            # 7. Statement timeout enforced (< 5 seconds)
            try:
                cur.execute("SELECT pg_sleep(6);")
                raise AssertionError("Security breach: analyst_ro exceeded 5s statement_timeout!")
            except psycopg.errors.QueryCanceled:
                print("  [OK] Statement timeout enforced on long queries (>5s)")


def main():
    print("=" * 60)
    print("Starting Text-to-SQL Database Verification")
    print("=" * 60)
    try:
        with get_owner_conn() as owner_conn:
            verify_row_counts(owner_conn)
            verify_profile(owner_conn)
        verify_security()
        print("=" * 60)
        print("ALL DATABASE VERIFICATIONS PASSED SUCCESSFULLY!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
