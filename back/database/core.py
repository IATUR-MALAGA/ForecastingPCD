import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence, Union

import psycopg
from psycopg.rows import dict_row

from back.config import settings

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": settings.get("db.connection.host", "localhost"),
    "port": int(settings.get("db.connection.port", 5432)),
    "dbname": settings.get("db.connection.dbname", ""),
    "user": settings.get("db.connection.user", ""),
    "password": settings.get("db.connection.password", ""),
    "connect_timeout": int(settings.get("db.connection.connect_timeout", 10)),
}


def create_connection_database() -> psycopg.Connection:
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
