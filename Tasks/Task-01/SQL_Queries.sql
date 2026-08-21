-- 1) Verify that all nine tables were loaded.
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL SELECT 'geolocation', COUNT(*) FROM geolocation
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'sellers', COUNT(*) FROM sellers
UNION ALL SELECT 'product_category_translation', COUNT(*) FROM product_category_translation
ORDER BY table_name;

-- 2) Required JOIN example: orders with their customers.
SELECT o.order_id, o.order_status, o.order_purchase_timestamp,
       c.customer_city, c.customer_state
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
ORDER BY o.order_purchase_timestamp DESC
LIMIT 10;

-- 3) Multi-table JOIN: order items, products, sellers and translated categories.
SELECT oi.order_id, oi.order_item_id, oi.price,
       COALESCE(t.product_category_name_english, p.product_category_name) AS category,
       s.seller_city, s.seller_state
FROM order_items AS oi
JOIN products AS p ON p.product_id = oi.product_id
JOIN sellers AS s ON s.seller_id = oi.seller_id
LEFT JOIN product_category_translation AS t
       ON t.product_category_name = p.product_category_name
LIMIT 10;

-- 4) Target definition for the future ML task.
-- Only delivered orders with known actual delivery dates are labeled.
SELECT order_id,
       CASE
         WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1
         ELSE 0
       END AS is_late
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
LIMIT 10;

-- 5) Sanity check: distribution of the future target (not full EDA).
SELECT CASE
         WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 'Late'
         ELSE 'On time'
       END AS delivery_class,
       COUNT(*) AS orders
FROM orders
WHERE order_status = 'delivered'
  AND order_delivered_customer_date IS NOT NULL
GROUP BY delivery_class
ORDER BY delivery_class;

