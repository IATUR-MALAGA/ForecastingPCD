import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence, Union

import psycopg
from psycopg.rows import dict_row

from back.config import settings

logger = logging.getLogger(__name__)

def _build_db_config() -> Dict[str, Any]:
    db_connection = settings.get("db.connection", {}) or {}

    db_config: Dict[str, Any] = {
        "host": db_connection.get("host", "localhost"),
        "port": int(db_connection.get("port", 5432)),
        "dbname": db_connection.get("dbname", ""),
        "user": db_connection.get("user", ""),
        "password": db_connection.get("password", ""),
        "connect_timeout": int(db_connection.get("connect_timeout", 10)),
    }
    return db_config


DB_CONFIG = _build_db_config()


def create_connection_database() -> psycopg.Connection:
    has_password = bool(DB_CONFIG.get("password"))
    logger.info(
        "Connecting to PostgreSQL host=%s port=%s dbname=%s user=%s has_password=%s",
        DB_CONFIG.get("host"),
        DB_CONFIG.get("port"),
        DB_CONFIG.get("dbname"),
        DB_CONFIG.get("user"),
        has_password,
    )

    if not has_password:
        raise RuntimeError(
            "DB password missing: expected db.connection.password in config.yml "
            f"(loaded from {settings.config_path})"
        )

    try:
        return psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    except psycopg.OperationalError:
        logger.exception("Failed to connect to PostgreSQL")
        raise


@contextmanager
def db_connection() -> Iterator[psycopg.Connection]:
    conn = create_connection_database()
    try:
        yield conn
    finally:
        conn.close()


Params = Union[Sequence[Any], Dict[str, Any], None]


def fetch_data(query, params: Params = None) -> List[Dict[str, Any]]:
    """
    Execute a SQL query and return all rows as dictionaries.
    query puede ser str o psycopg.sql.Composed/SQL.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            if params is None:
                cur.execute(query)
            else:
                cur.execute(query, params)
            return cur.fetchall()
