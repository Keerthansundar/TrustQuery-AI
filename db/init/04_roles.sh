#!/usr/bin/env bash
set -e

ANALYST_RO_PASS="${ANALYST_RO_PASSWORD:-change_this_analyst_ro_password}"
ANALYST_PII_RO_PASS="${ANALYST_PII_RO_PASSWORD:-change_this_analyst_pii_ro_password}"
APP_RW_PASS="${APP_RW_PASSWORD:-change_this_app_rw_password}"

echo "Configuring database roles and security privileges..."

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- 1. Create analyst_ro role
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'analyst_ro') THEN
            CREATE ROLE analyst_ro WITH LOGIN PASSWORD '$ANALYST_RO_PASS' CONNECTION LIMIT 20;
        ELSE
            ALTER ROLE analyst_ro WITH PASSWORD '$ANALYST_RO_PASS' CONNECTION LIMIT 20;
        END IF;
    END
    \$\$;

    -- Revoke everything first for safety
    REVOKE ALL ON SCHEMA public, rag, app FROM analyst_ro;
    REVOKE ALL ON ALL TABLES IN SCHEMA public, rag, app FROM analyst_ro;

    -- Grant usage on public schema
    GRANT USAGE ON SCHEMA public TO analyst_ro;

    -- Grant select on all business tables except customers
    GRANT SELECT ON public.categories, public.products, public.orders, public.order_items, public.payments, public.shipments TO analyst_ro;

    -- Column-level grant on customers: EXCLUDE email and phone (PII)
    GRANT SELECT (customer_id, first_name, last_name, city, state, country, signup_date, segment) ON public.customers TO analyst_ro;

    -- Security parameters: read-only, strict timeouts
    ALTER ROLE analyst_ro SET default_transaction_read_only = on;
    ALTER ROLE analyst_ro SET statement_timeout = '5s';
    ALTER ROLE analyst_ro SET lock_timeout = '2s';
    ALTER ROLE analyst_ro SET idle_in_transaction_session_timeout = '10s';

    -- 2. Create analyst_pii_ro role
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'analyst_pii_ro') THEN
            CREATE ROLE analyst_pii_ro WITH LOGIN PASSWORD '$ANALYST_PII_RO_PASS' CONNECTION LIMIT 20;
        ELSE
            ALTER ROLE analyst_pii_ro WITH PASSWORD '$ANALYST_PII_RO_PASS' CONNECTION LIMIT 20;
        END IF;
    END
    \$\$;

    REVOKE ALL ON SCHEMA public, rag, app FROM analyst_pii_ro;
    REVOKE ALL ON ALL TABLES IN SCHEMA public, rag, app FROM analyst_pii_ro;

    GRANT USAGE ON SCHEMA public TO analyst_pii_ro;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_pii_ro;

    ALTER ROLE analyst_pii_ro SET default_transaction_read_only = on;
    ALTER ROLE analyst_pii_ro SET statement_timeout = '5s';
    ALTER ROLE analyst_pii_ro SET lock_timeout = '2s';
    ALTER ROLE analyst_pii_ro SET idle_in_transaction_session_timeout = '10s';

    -- 3. Create app_rw role (only rag and app schemas)
    DO \$\$
    BEGIN
        IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'app_rw') THEN
            CREATE ROLE app_rw WITH LOGIN PASSWORD '$APP_RW_PASS' CONNECTION LIMIT 20;
        ELSE
            ALTER ROLE app_rw WITH PASSWORD '$APP_RW_PASS' CONNECTION LIMIT 20;
        END IF;
    END
    \$\$;

    REVOKE ALL ON SCHEMA public, rag, app FROM app_rw;
    REVOKE ALL ON ALL TABLES IN SCHEMA public, rag, app FROM app_rw;

    GRANT USAGE ON SCHEMA rag, app TO app_rw;
    GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rag, app TO app_rw;
    GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rag, app TO app_rw;
    ALTER DEFAULT PRIVILEGES IN SCHEMA rag GRANT ALL ON TABLES TO app_rw;
    ALTER DEFAULT PRIVILEGES IN SCHEMA app GRANT ALL ON TABLES TO app_rw;
EOSQL

echo "Roles configured successfully."
