"""Database utilities for interacting with PostgreSQL database.

This module provides utilities for connecting to and interacting with a PostgreSQL
database, with a focus on efficient batch operations and proper connection management.
"""

import logging
import os
from contextlib import contextmanager
from typing import Generator, Literal

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

logger = logging.getLogger(__name__)


def create_db_engine() -> Engine:
    """Create SQLAlchemy engine with connection pooling.

    Returns
    -------
    Engine
        SQLAlchemy engine instance configured with connection pooling

    Raises
    ------
    ValueError
        If DATABASE_URL environment variable is not set
    SQLAlchemyError
        If connection to database fails
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable must be set")

    try:
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_pre_ping=True,
        )
        return engine
    except SQLAlchemyError as e:
        logger.error("Failed to create database engine: %s", str(e))
        raise


@contextmanager
def db_connection(engine: Engine) -> Generator[Connection, None, None]:
    """Context manager for database connections.

    Parameters
    ----------
    engine : Engine
        SQLAlchemy engine instance

    Yields
    ------
    Connection
        Database connection object

    Raises
    ------
    SQLAlchemyError
        If database operations fail
    """
    try:
        connection = engine.connect()
        yield connection
    except SQLAlchemyError as e:
        logger.error("Database operation failed: %s", str(e))
        raise
    finally:
        connection.close()


def write_dataframe_to_db(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    if_exists: Literal["fail", "replace", "append"] = "replace",
    chunk_size: int = 10000,
) -> None:
    """Write DataFrame to PostgreSQL database efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write to database
    table_name : str
        Name of the target table
    engine : Engine
        SQLAlchemy engine instance
    if_exists : Literal["fail", "replace", "append"], optional
        How to behave if table exists, by default "replace"
    chunk_size : int, optional
        Number of rows to write at once, by default 10000

    Raises
    ------
    SQLAlchemyError
        If writing to database fails
    """
    try:
        with db_connection(engine) as conn:
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                index=False,
                chunksize=chunk_size,
            )
            logger.info(f"Successfully wrote {len(df):,} rows to table {table_name}")
    except SQLAlchemyError as e:
        logger.error(f"Failed to write data to table {table_name}: {str(e)}")
        raise
