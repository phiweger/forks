from datetime import datetime
from typing import Dict, Optional
from uuid import uuid4

# from pydantic import BaseModel, Field
from sqlmodel import SQLModel, Column, Session, select, JSON, Field


def get_datetime():
    # Milliseconds.
    # https://stackoverflow.com/questions/7588511/format-a-datetime-into-a-string-with-milliseconds
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


class Memory(SQLModel, table=True):
    """
    Memory (for now) is organized in threads. Think of it as a single conversation, a single thread in a forum, a single train of thought.

    Create table: Only run once -- https://sqlmodel.tiangolo.com/tutorial/create-db-and-table/

    import os
    from dotenv import load_dotenv
    from sqlmodel import Session, Field, SQLModel, create_engine

    load_dotenv()
    uri = "postgresql://" + os.environ["DATABASE_URI"]
    engine = create_engine(uri, echo=True)
    SQLModel.metadata.create_all(engine)
    """

    message_id: str = Field(..., primary_key=True)
    thread_id: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    timestamp: str = Field(
        default_factory=get_datetime
    )  # datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    content: Optional[Dict] = Field(default={}, sa_column=Column(JSON))
    # For example: {"role": "adventurer", "content": "Ist es ein Makrolid?"}
    # Format from ChatGPT.

    # Needed for Column(JSON)
    class Config:
        arbitrary_types_allowed = True


class MemoryStream:
    """
    Manages memory communication with the database.

    An object that represents a single memory thread.
    """

    def __init__(self, engine=None, user_id=None, thread_id=None, size=100):
        self.user_id = user_id
        self.thread_id = thread_id
        self.size = size
        self.cache = []  # short term memory
        self.engine = engine

    def get_history(self):
        with Session(self.engine) as session:
            statement = select(Memory).where(
                Memory.user_id == self.user_id, Memory.thread_id == self.thread_id
            )
            results = session.exec(statement).fetchall()
            # import pdb; pdb.set_trace()
            history = [i.content for i in results]
        self.cache = history
        return history

    def add_message(self, content, force=False):
        """
        Persist memory and add message to cache.

        content .. Any dict, will be dumped in JSON field in the database table.
        """
        if (len(self.cache) + 1 > self.size) and not force:
            return False

        else:
            id_ = uuid4().__str__()
            db_record = {
                "user_id": self.user_id,
                "thread_id": self.thread_id,
                "message_id": id_,
                "content": content,
            }
            m = Memory(**db_record)
            with Session(self.engine) as session:
                session.add(m)
                session.commit()
            self.cache.append(content)
            return id_
