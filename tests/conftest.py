from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture()
def sample_database(tmp_path: Path) -> Path:
    path = tmp_path / "shop.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL,
                total REAL,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            INSERT INTO customers VALUES (1, 'Ada'), (2, 'Lin');
            INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 20.0), (3, 2, NULL);
            """
        )
    return path

