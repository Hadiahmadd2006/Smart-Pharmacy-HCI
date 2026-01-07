from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading

_DB_LOCK = threading.Lock()
_DB_CONN: sqlite3.Connection | None = None
_SQL_PATH: Path | None = None


DEFAULT_SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    role TEXT NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS faces_unique ON faces(name, role);

CREATE TABLE IF NOT EXISTS inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    stock INTEGER NOT NULL,
    price REAL NOT NULL,
    marker_id INTEGER,
    image_path TEXT,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS inventory_unique ON inventory(name);

CREATE TABLE IF NOT EXISTS purchases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    items_json TEXT NOT NULL,
    total_qty INTEGER NOT NULL,
    total_price REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heatmaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    image_path TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS heatmaps_unique ON heatmaps(customer_name, kind);

CREATE TABLE IF NOT EXISTS gaze_calibration (
    customer_name TEXT PRIMARY KEY,
    feature_map TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
""".strip()


def sql_escape(value: str) -> str:
    return value.replace("'", "''")


def init_database(sql_path: Path) -> sqlite3.Connection:
    global _DB_CONN, _SQL_PATH
    _SQL_PATH = sql_path
    sql_path.parent.mkdir(parents=True, exist_ok=True)

    if _DB_CONN is not None:
        return _DB_CONN

    _DB_CONN = sqlite3.connect(":memory:", check_same_thread=False)
    _DB_CONN.execute("PRAGMA foreign_keys = ON;")

    if sql_path.exists():
        _DB_CONN.executescript(sql_path.read_text(encoding="utf-8"))
    else:
        _DB_CONN.executescript(DEFAULT_SCHEMA)
        export_database_sql(sql_path)
        return _DB_CONN
    _DB_CONN.executescript(DEFAULT_SCHEMA)
    _ensure_purchase_columns(_DB_CONN)
    return _DB_CONN


def get_connection() -> sqlite3.Connection:
    if _DB_CONN is None:
        raise RuntimeError("Database is not initialized.")
    return _DB_CONN


@contextmanager
def db_session() -> sqlite3.Connection:
    conn = get_connection()
    with _DB_LOCK:
        yield conn
        conn.commit()


def export_database_sql(sql_path: Path | None = None) -> None:
    if sql_path is None:
        if _SQL_PATH is None:
            raise RuntimeError("SQL path is not set.")
        sql_path = _SQL_PATH

    lines = [
        "-- Editable database (faces, inventory, purchases, heatmaps, gaze calibration)",
        DEFAULT_SCHEMA,
        "DELETE FROM faces;",
        "DELETE FROM inventory;",
        "DELETE FROM purchases;",
        "DELETE FROM heatmaps;",
        "DELETE FROM gaze_calibration;",
    ]

    with _DB_LOCK:
        conn = get_connection()
        cursor = conn.execute(
            "SELECT name, role, embedding, dim, model, created_at FROM faces ORDER BY name"
        )
        for name, role, embedding_blob, dim, model, created_at in cursor.fetchall():
            hex_blob = embedding_blob.hex() if embedding_blob else ""
            lines.append(
                "INSERT INTO faces (name, role, embedding, dim, model, created_at) "
                f"VALUES ('{sql_escape(name)}', '{sql_escape(role)}', "
                f"X'{hex_blob}', {int(dim)}, '{sql_escape(model)}', '{sql_escape(created_at)}');"
            )

        cursor = conn.execute(
            "SELECT name, stock, price, marker_id, image_path, updated_at FROM inventory ORDER BY name"
        )
        for name, stock, price, marker_id, image_path, updated_at in cursor.fetchall():
            marker = "NULL" if marker_id is None else str(int(marker_id))
            lines.append(
                "INSERT INTO inventory (name, stock, price, marker_id, image_path, updated_at) "
                f"VALUES ('{sql_escape(name)}', {int(stock)}, {float(price)}, "
                f"{marker}, '{sql_escape(image_path or '')}', '{sql_escape(updated_at)}');"
            )

        cursor = conn.execute(
            "SELECT customer_name, items_json, total_qty, total_price, created_at FROM purchases ORDER BY id"
        )
        for customer_name, items_json, total_qty, total_price, created_at in cursor.fetchall():
            lines.append(
                "INSERT INTO purchases (customer_name, items_json, total_qty, total_price, created_at) "
                f"VALUES ('{sql_escape(customer_name)}', '{sql_escape(items_json)}', "
                f"{int(total_qty)}, {float(total_price)}, '{sql_escape(created_at)}');"
            )

        cursor = conn.execute(
            "SELECT customer_name, kind, image_path, updated_at FROM heatmaps ORDER BY customer_name, kind"
        )
        for customer_name, kind, image_path, updated_at in cursor.fetchall():
            lines.append(
                "INSERT INTO heatmaps (customer_name, kind, image_path, updated_at) "
                f"VALUES ('{sql_escape(customer_name)}', '{sql_escape(kind)}', "
                f"'{sql_escape(image_path)}', '{sql_escape(updated_at)}');"
            )

        cursor = conn.execute(
            "SELECT customer_name, feature_map, updated_at FROM gaze_calibration ORDER BY customer_name"
        )
        for customer_name, feature_map, updated_at in cursor.fetchall():
            lines.append(
                "INSERT INTO gaze_calibration (customer_name, feature_map, updated_at) "
                f"VALUES ('{sql_escape(customer_name)}', '{sql_escape(feature_map)}', "
                f"'{sql_escape(updated_at)}');"
            )

    sql_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_purchase_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(purchases)")}
    if "total_price" not in columns:
        conn.execute(
            "ALTER TABLE purchases ADD COLUMN total_price REAL NOT NULL DEFAULT 0.0"
        )
