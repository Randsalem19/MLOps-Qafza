"""Stream the nine Olist CSV files from a ZIP archive into PostgreSQL."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

FILES_TO_TABLES = [
    ("olist_customers_dataset.csv", "customers"),
    ("olist_geolocation_dataset.csv", "geolocation"),
    ("olist_sellers_dataset.csv", "sellers"),
    ("olist_products_dataset.csv", "products"),
    ("product_category_name_translation.csv", "product_category_translation"),
    ("olist_orders_dataset.csv", "orders"),
    ("olist_order_items_dataset.csv", "order_items"),
    ("olist_order_payments_dataset.csv", "order_payments"),
    ("olist_order_reviews_dataset.csv", "order_reviews"),
]


def connection_string() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB', 'olist')} "
        f"user={os.getenv('POSTGRES_USER', 'olist_user')} "
        f"password={os.getenv('POSTGRES_PASSWORD', '')}"
    )


def find_zip() -> Path:
    configured = os.getenv("DATA_ZIP")
    candidates = [Path(configured)] if configured else []
    candidates.extend(Path("data").glob("*.zip"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Dataset ZIP not found. Put it inside data/ or set the DATA_ZIP variable."
    )


def copy_csv(conn: psycopg.Connection, archive: zipfile.ZipFile, csv_name: str, table: str) -> int:
    with archive.open(csv_name, "r") as source, conn.cursor() as cursor:
        with cursor.copy(f"COPY {table} FROM STDIN WITH (FORMAT CSV, HEADER TRUE)") as copy:
            while chunk := source.read(1024 * 1024):
                copy.write(chunk)
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]


def main() -> None:
    archive_path = find_zip()
    print(f"Using dataset: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        missing = [name for name, _ in FILES_TO_TABLES if name not in archive.namelist()]
        if missing:
            raise ValueError(f"Missing CSV files in archive: {missing}")

        with psycopg.connect(connection_string()) as conn:
            with conn.cursor() as cursor:
                tables = ", ".join(table for _, table in reversed(FILES_TO_TABLES))
                cursor.execute(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE")
            conn.commit()

            for csv_name, table in FILES_TO_TABLES:
                rows = copy_csv(conn, archive, csv_name, table)
                conn.commit()
                print(f"Loaded {table:<30} {rows:>10,} rows")

    print("All Olist tables were loaded successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Load failed: {exc}", file=sys.stderr)
        raise
