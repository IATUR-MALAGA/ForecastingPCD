import logging
from typing import Any, Dict, List, Sequence, Union, Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host": "alcazaba2.uma.es",
    "port": 5432,
    "dbname": "unibigdata",
    "user": "unibigdata",
    "password": "unibigdata",
}

def create_connection_database() -> psycopg.Connection:
    """
    Create and return a PostgreSQL connection.

    Raises
    ------
    psycopg.OperationalError
        If the connection cannot be established.
    """
    try:
        return psycopg.connect(**DB_CONFIG, row_factory=dict_row)
    except psycopg.OperationalError:
        logger.exception("Failed to connect to PostgreSQL")
        raise


@contextmanager
def db_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a PostgreSQL connection and always close it.
    """
    conn = create_connection_database()
    try:
        yield conn
    finally:
        conn.close()


Params = Union[Sequence[Any], Dict[str, Any], None]

def fetch_data(query: str, params: Params = None) -> List[Dict[str, Any]]:
    """
    Execute a SQL query and return all rows as dictionaries.

    Raises
    ------
    psycopg.Error
        If query execution fails.
    """
    with db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()

