"""Deterministic SQL Guardrail using sqlglot AST parsing (dialect='postgres')."""

import logging

import sqlglot
from sqlglot import exp

from src.config import settings

logger = logging.getLogger(__name__)

# Allowed tables in schema public
ALLOWED_TABLES: set[str] = {
    "categories",
    "customers",
    "products",
    "orders",
    "order_items",
    "payments",
    "shipments",
}

# Complete schema catalog of allowed columns per table
SCHEMA_CATALOG: dict[str, set[str]] = {
    "categories": {"category_id", "category_name", "parent_category_id"},
    "customers": {
        "customer_id",
        "first_name",
        "last_name",
        "email",
        "phone",
        "city",
        "state",
        "country",
        "signup_date",
        "segment",
    },
    "products": {
        "product_id",
        "product_name",
        "category_id",
        "brand",
        "unit_price",
        "cost_price",
        "created_at",
        "is_active",
    },
    "orders": {
        "order_id",
        "customer_id",
        "order_date",
        "status",
        "channel",
        "shipping_fee",
    },
    "order_items": {
        "order_item_id",
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_pct",
    },
    "payments": {
        "payment_id",
        "order_id",
        "payment_date",
        "payment_method",
        "amount",
        "payment_status",
    },
    "shipments": {
        "shipment_id",
        "order_id",
        "carrier",
        "shipped_date",
        "delivered_date",
        "shipment_status",
    },
}

# Denied PII columns unless ALLOW_PII is True
PII_COLUMNS: set[str] = {"email", "phone"}

# Allowed functions in anonymous AST nodes (standard functions like count/sum/avg are concrete classes)
ALLOWED_ANONYMOUS_FUNCTIONS: set[str] = {
    "round",
    "abs",
    "ceil",
    "ceiling",
    "floor",
    "power",
    "sqrt",
    "coalesce",
    "nullif",
    "greatest",
    "least",
    "concat",
    "lower",
    "upper",
    "trim",
    "btrim",
    "ltrim",
    "rtrim",
    "substr",
    "substring",
    "replace",
    "length",
    "left",
    "right",
    "split_part",
    "cast",
    "date_trunc",
    "extract",
    "date",
    "now",
    "current_date",
    "age",
    "to_date",
    "to_char",
    "to_timestamp",
    "dense_rank",
    "rank",
    "row_number",
    "lag",
    "lead",
    "cume_dist",
    "percent_rank",
    "ntile",
}

# Write and DDL AST node types that represent WRITE_ATTEMPT
WRITE_AST_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Command,
    exp.Set,
)


def sql_guard(sql: str, allow_pii: bool | None = None) -> tuple[bool, str, str, str]:
    """Validates and sanitizes a SQL query using AST inspection.

    Returns:
        (ok: bool, code: str, reason: str, sanitized_sql: str)
    """
    pii_allowed = allow_pii if allow_pii is not None else settings.ALLOW_PII

    if not sql or not sql.strip():
        return False, "PARSE_ERROR", "SQL query cannot be empty.", ""

    clean_sql = sql.strip().rstrip(";")

    # 1. Parse statement count
    try:
        statements = sqlglot.parse(clean_sql, read="postgres")
    except Exception as exc:
        return False, "PARSE_ERROR", f"SQL syntax error: {exc}", clean_sql

    if not statements:
        return False, "PARSE_ERROR", "No executable SQL statements found.", clean_sql

    if len(statements) > 1:
        return False, "WRITE_ATTEMPT", "Multiple statements are not permitted.", clean_sql

    root = statements[0]
    if root is None:
        return False, "PARSE_ERROR", "Unable to parse SQL query.", clean_sql

    # 2. Check for explicit write/DDL nodes at any level (including inside CTEs)
    for write_type in WRITE_AST_NODES:
        if isinstance(root, write_type) or list(root.find_all(write_type)):
            return False, "WRITE_ATTEMPT", "I can only run read-only queries.", clean_sql

    # Root must be a SELECT (or Union of Selects)
    if not isinstance(root, (exp.Select, exp.Union)):
        return False, "WRITE_ATTEMPT", "I can only run read-only queries.", clean_sql

    # 3. Check for SELECT ... INTO or locking clauses (FOR UPDATE / FOR SHARE)
    if root.args.get("into"):
        return False, "WRITE_ATTEMPT", "SELECT ... INTO write queries are forbidden.", clean_sql

    if root.args.get("locks"):
        return False, "WRITE_ATTEMPT", "FOR UPDATE/SHARE locking clauses are forbidden.", clean_sql

    # Collect CTE aliases and output column aliases
    cte_aliases = {cte.alias.lower() for cte in root.ctes if cte.alias}
    cte_columns = set()
    for cte in root.ctes:
        if isinstance(cte.this, exp.Select):
            for sel in cte.this.selects:
                if sel.alias:
                    cte_columns.add(sel.alias.lower())
                elif isinstance(sel, exp.Column):
                    cte_columns.add(sel.name.lower())

    # Collect all SELECT aliases in the query
    select_aliases = {
        alias_node.alias.lower() for alias_node in root.find_all(exp.Alias) if alias_node.alias
    }

    # 4. Check tables against table allowlist
    found_tables = set()
    for table_expr in root.find_all(exp.Table):
        tname = table_expr.name.lower()
        tschema = table_expr.db.lower() if table_expr.db else "public"

        # If it's a CTE alias, ignore
        if tname in cte_aliases:
            continue

        # Reject system schemas
        if tschema in ("pg_catalog", "information_schema", "rag", "app") or tschema.startswith(
            "pg_"
        ):
            return False, "UNKNOWN_TABLE", f"Access to schema '{tschema}' is forbidden.", clean_sql

        if tname.startswith("pg_"):
            return (
                False,
                "UNKNOWN_TABLE",
                f"Access to system table '{tname}' is forbidden.",
                clean_sql,
            )

        if tname not in ALLOWED_TABLES:
            return (
                False,
                "UNKNOWN_TABLE",
                f"Unknown table '{tname}'. Only public business tables are permitted: {sorted(ALLOWED_TABLES)}.",
                clean_sql,
            )

        found_tables.add(tname)

    # 5. Check functions against function allowlist
    for func in root.find_all(exp.Anonymous):
        fname = func.name.lower()
        if fname not in ALLOWED_ANONYMOUS_FUNCTIONS:
            return (
                False,
                "FORBIDDEN_FUNCTION",
                f"Function '{fname}' is forbidden. Only standard safe analytical functions are allowed.",
                clean_sql,
            )

    # 6. Check for disallowed PII columns and unknown columns
    all_allowed_columns = set()
    for t in found_tables:
        all_allowed_columns.update(SCHEMA_CATALOG.get(t, set()))

    for col in root.find_all(exp.Column):
        cname = col.name.lower()

        # If column belongs to a CTE, skip column schema verification
        if col.table and col.table.lower() in cte_aliases:
            continue

        # Check PII denylist
        if not pii_allowed and cname in PII_COLUMNS:
            return (
                False,
                "PII_ACCESS_DENIED",
                f"Access to PII column '{cname}' is denied. Please remove it from the query.",
                clean_sql,
            )

        # Check unknown column
        if col.table and col.table.lower() in SCHEMA_CATALOG:
            valid_cols = SCHEMA_CATALOG[col.table.lower()]
            if cname not in valid_cols and cname != "*":
                return (
                    False,
                    "UNKNOWN_COLUMN",
                    f"Unknown column '{cname}' in table '{col.table.lower()}'.",
                    clean_sql,
                )
        elif not col.table and found_tables:
            # Unqualified column
            if (
                cname not in all_allowed_columns
                and cname not in cte_columns
                and cname not in select_aliases
                and cname != "*"
            ):
                return (
                    False,
                    "UNKNOWN_COLUMN",
                    f"Unknown column '{cname}'.",
                    clean_sql,
                )

    
    # 7. Expand standalone SELECT * if present.
    # Do NOT treat COUNT(*) as SELECT *.
    modified = False

    select_star_projections = []

    for select_node in root.find_all(exp.Select):
        for projection in select_node.expressions:
            expression = (
                projection.this
                if isinstance(projection, exp.Alias)
                else projection
            )

            if isinstance(expression, exp.Star):
                select_star_projections.append((select_node, projection))
            elif (
                isinstance(expression, exp.Column)
                and expression.is_star
            ):
                select_star_projections.append((select_node, projection))

    if select_star_projections:
        has_grouping = any(
            select_node.args.get("group")
            for select_node in root.find_all(exp.Select)
        )
        has_aggregation = bool(list(root.find_all(exp.AggFunc)))

        if has_grouping or has_aggregation:
            return (
                False,
                "INVALID_AGGREGATION",
                (
                    "SELECT * is not allowed in aggregate or grouped queries. "
                    "Select only the grouping columns and required aggregate "
                    "expressions, with appropriate aliases."
                ),
                clean_sql,
            )

        if len(found_tables) == 1:
            table_name = next(iter(found_tables))

            permitted = [
                c for c in sorted(SCHEMA_CATALOG[table_name])
                if pii_allowed or c not in PII_COLUMNS
            ]

            replacement_exprs = [
                sqlglot.to_identifier(col_name)
                for col_name in permitted
            ]

            for select_node, _ in select_star_projections:
                select_node.set("expressions", replacement_exprs)
            
            modified = True

            logger.info(
                "Expanded SELECT * to explicit permitted columns: %s",
                permitted,
            )


    # 8. Enforce and clamp LIMIT
    is_aggregate = bool(list(root.find_all(exp.AggFunc))) and not root.args.get("group")
    limit_node = root.args.get("limit")

    if limit_node is None:
        if not is_aggregate:
            root = root.limit(settings.MAX_ROWS)
            modified = True
    else:
        # If limit is specified, ensure it does not exceed MAX_ROWS
        try:
            val_node = limit_node.expression
            if isinstance(val_node, exp.Literal):
                limit_val = int(val_node.this)
                if limit_val > settings.MAX_ROWS:
                    root.set("limit", exp.Limit(expression=exp.Literal.number(settings.MAX_ROWS)))
                    modified = True
        except Exception:
            pass

    sanitized_sql = root.sql(dialect="postgres") if modified else clean_sql
    return True, "OK", "", sanitized_sql
