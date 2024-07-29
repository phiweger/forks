"""
Saving session state in FastAPI seems tricky? Apparently we need to use middleware -- https://stackoverflow.com/questions/70617258/session-object-in-fastapi-similar-to-flask

> A "middleware" is a function that works with every request before it is processed by any specific path operation. And also with every response before returning it. -- https://fastapi.tiangolo.com/tutorial/middleware/

See also:

- https://github.com/tiangolo/fastapi/issues/4746
- https://github.com/tiangolo/fastapi/discussions/6358
- Deprecated: https://jordanisaacs.github.io/fastapi-sessions/

However, for now we want to keep all data, so we'll just use the database.
"""

from datetime import datetime, timezone
from typing import Optional, Dict

from loguru import logger
from sqlmodel import SQLModel, Field, Column, JSON, Session, text


def get_timestamp():
    """
    A callable for the default factory of the timestamp field.
    """
    return datetime.now(tz=timezone.utc)


def create_table_if_not_exists(engine, tables=[]):
    """
    https://stackoverflow.com/questions/17044259/how-to-check-if-table-exists

    Usage:

    assert create_table_if_not_exists([Queries, Citations, Vote, QueryExecutionTimes])
    """
    if not tables:
        raise ValueError("No tables provided.")

    success = []
    with Session(engine) as session:
        for table in tables:
            table_name = table.__tablename__
            logger.info(f"Checking if table {table_name} exists ...")
            statement = text(
                f"""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = '{table_name}'
                """
            )
            results = session.exec(statement)
            x = results.fetchone()
            logger.info(x)
            if x[0] == 1:
                logger.info(f"Table {table_name} already exists.")
            else:
                logger.info(f"Table {table_name} does not exists, creating ...")
                # Create (single) table.
                # SQLModel.metadata.create_all(engine)
                # https://stackoverflow.com/questions/73631480/how-to-create-table-in-existing-database-with-sqlmodel
                table.__table__.create(engine)
            success.append(True)

    logger.info(f"Tables created: {success}")
    return all(success)


class GraphState(SQLModel, table=True):
    """
    Graph traversal session.
    """

    # Session ID cannot be primary key because we want to add a new record for each routing.
    record_id: int = Field(default=None, primary_key=True)
    graph_name: str | None = Field(default=None)
    session_id: str = Field(...)  # required
    user_id: str | None = Field(default=None)
    # Timestamp:
    # https://github.com/tiangolo/sqlmodel/issues/594
    # https://stackoverflow.com/questions/3327946/how-can-i-get-the-current-time-now-in-utc
    timestamp: datetime = Field(default_factory=get_timestamp)
    current_node: str | None = Field(default=None)
    response: str | None = Field(default=None)
    route_to: str | None = Field(default=None)
    # A field to dump more information.
    comment: Optional[Dict] = Field(default={}, sa_column=Column(JSON))
    # For example: {"role": "adventurer", "content": "Ist es ein Makrolid?"}
    # Format from ChatGPT.

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True
