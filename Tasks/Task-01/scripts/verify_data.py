"""Verify expected row counts and a representative JOIN."""

import psycopg

from load_data import connection_string

EXPECTED = {
    "customers": 99_441,
    "geolocation": 1_000_163,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "orders": 99_441,
    "products": 32_951,
    "sellers": 3_095,
    "product_category_translation": 71,
}

with psycopg.connect(connection_string()) as conn, conn.cursor() as cursor:
    for table, expected in EXPECTED.items():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cursor.fetchone()[0]
        assert actual == expected, f"{table}: expected {expected}, found {actual}"
        print(f"PASS {table:<30} {actual:>10,}")

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        """
    )
    joined = cursor.fetchone()[0]
    assert joined == EXPECTED["orders"]
    print(f"PASS orders-customers JOIN         {joined:>10,}")

print("Database verification completed successfully.")

