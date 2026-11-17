-- ==============================================================================
-- 02_load.sql: Server-Side Data Load from /data/raw in Dependency Order
-- ==============================================================================

\echo 'Loading categories...'
COPY public.categories (category_id, category_name, parent_category_id)
FROM '/data/raw/categories.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading customers...'
COPY public.customers (customer_id, first_name, last_name, email, phone, city, state, country, signup_date, segment)
FROM '/data/raw/customers.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading products...'
COPY public.products (product_id, product_name, category_id, brand, unit_price, cost_price, created_at, is_active)
FROM '/data/raw/products.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading orders...'
COPY public.orders (order_id, customer_id, order_date, status, channel, shipping_fee)
FROM '/data/raw/orders.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading order_items...'
COPY public.order_items (order_item_id, order_id, product_id, quantity, unit_price, discount_pct)
FROM '/data/raw/order_items.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading payments...'
COPY public.payments (payment_id, order_id, payment_date, payment_method, amount, payment_status)
FROM '/data/raw/payments.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Loading shipments...'
COPY public.shipments (shipment_id, order_id, carrier, shipped_date, delivered_date, shipment_status)
FROM '/data/raw/shipments.csv'
WITH (FORMAT csv, HEADER true);

\echo 'Analyzing database tables...'
ANALYZE;
