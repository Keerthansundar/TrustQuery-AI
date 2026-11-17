-- ==============================================================================
-- 01_schema.sql: Extensions, Schemas, Business Tables, Constraints, and Indexes
-- ==============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Ensure required schemas exist
CREATE SCHEMA IF NOT EXISTS public;
CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS app;

-- ------------------------------------------------------------------------------
-- Table 1: categories
-- Two levels: 5 top-level (parent NULL) and 10 leaf categories.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.categories (
    category_id INT PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE,
    parent_category_id INT NULL REFERENCES public.categories(category_id)
);

CREATE INDEX IF NOT EXISTS idx_categories_parent_category_id ON public.categories(parent_category_id);

-- ------------------------------------------------------------------------------
-- Table 2: customers
-- email and phone are PII
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.customers (
    customer_id INT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT NOT NULL UNIQUE,
    phone TEXT,
    city TEXT,
    state TEXT,
    country TEXT,
    signup_date DATE,
    segment TEXT CHECK (segment IN ('consumer', 'corporate', 'small_business'))
);

CREATE INDEX IF NOT EXISTS idx_customers_signup_date ON public.customers(signup_date);
CREATE INDEX IF NOT EXISTS idx_customers_segment ON public.customers(segment);

-- ------------------------------------------------------------------------------
-- Table 3: products
-- References leaf categories only. CSV holds 1/0 for is_active.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.products (
    product_id INT PRIMARY KEY,
    product_name TEXT,
    category_id INT REFERENCES public.categories(category_id),
    brand TEXT,
    unit_price NUMERIC(10,2),
    cost_price NUMERIC(10,2) CHECK (cost_price < unit_price),
    created_at DATE,
    is_active BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_products_category_id ON public.products(category_id);
CREATE INDEX IF NOT EXISTS idx_products_created_at ON public.products(created_at);
CREATE INDEX IF NOT EXISTS idx_products_brand ON public.products(brand);

-- ------------------------------------------------------------------------------
-- Table 4: orders
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.orders (
    order_id INT PRIMARY KEY,
    customer_id INT REFERENCES public.customers(customer_id),
    order_date DATE,
    status TEXT CHECK (status IN ('placed', 'shipped', 'delivered', 'cancelled', 'returned')),
    channel TEXT CHECK (channel IN ('web', 'mobile_app', 'store')),
    shipping_fee NUMERIC(8,2)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON public.orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_order_date ON public.orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON public.orders(status);

-- ------------------------------------------------------------------------------
-- Table 5: order_items
-- unit_price is at purchase time, quantity > 0, discount_pct between 0 and 30
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.order_items (
    order_item_id INT PRIMARY KEY,
    order_id INT REFERENCES public.orders(order_id),
    product_id INT REFERENCES public.products(product_id),
    quantity INT CHECK (quantity > 0),
    unit_price NUMERIC(10,2),
    discount_pct NUMERIC(5,2) CHECK (discount_pct >= 0 AND discount_pct <= 30)
);

CREATE INDEX IF NOT EXISTS idx_order_items_order_id ON public.order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_product_id ON public.order_items(product_id);

-- ------------------------------------------------------------------------------
-- Table 6: payments
-- One payment per order. amount = items total + shipping_fee
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.payments (
    payment_id INT PRIMARY KEY,
    order_id INT UNIQUE REFERENCES public.orders(order_id),
    payment_date DATE,
    payment_method TEXT CHECK (payment_method IN ('credit_card', 'upi', 'paypal', 'bank_transfer', 'cod')),
    amount NUMERIC(12,2),
    payment_status TEXT CHECK (payment_status IN ('paid', 'failed', 'refunded', 'pending'))
);

CREATE INDEX IF NOT EXISTS idx_payments_order_id ON public.payments(order_id);
CREATE INDEX IF NOT EXISTS idx_payments_payment_date ON public.payments(payment_date);
CREATE INDEX IF NOT EXISTS idx_payments_payment_status ON public.payments(payment_status);

-- ------------------------------------------------------------------------------
-- Table 7: shipments
-- One shipment per shipped/delivered/returned order. delivered_date can be NULL.
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.shipments (
    shipment_id INT PRIMARY KEY,
    order_id INT UNIQUE REFERENCES public.orders(order_id),
    carrier TEXT,
    shipped_date DATE,
    delivered_date DATE NULL,
    shipment_status TEXT CHECK (shipment_status IN ('in_transit', 'delivered', 'returned'))
);

CREATE INDEX IF NOT EXISTS idx_shipments_order_id ON public.shipments(order_id);
CREATE INDEX IF NOT EXISTS idx_shipments_shipped_date ON public.shipments(shipped_date);
CREATE INDEX IF NOT EXISTS idx_shipments_delivered_date ON public.shipments(delivered_date);
CREATE INDEX IF NOT EXISTS idx_shipments_shipment_status ON public.shipments(shipment_status);
