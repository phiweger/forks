from pydantic import BaseModel


class Query(BaseModel):
    graph: str
    user_id: str
    session_id: str | None
    query: str | None
