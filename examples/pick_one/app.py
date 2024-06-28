from fastapi import FastAPI
from pydantic import BaseModel, Field

from forks.switch import AsyncSwitch, vote


app = FastAPI()


class Query(BaseModel):
    query: str
    states: dict
    comment: str | None = Field(
        None, description="A comment to help the switch make a decision."
    )
    method: str = Field(
        "logit_bias",
        description="The method to use for the switch.",
        enum=["logit_bias", "fn_call"],
    )
    model: str
    vote: bool = Field(
        False,
        description="Call the switch multiple times and return the most likely state on average.",
    )


class Decision(BaseModel):
    response: str | int


@app.post("/switch", response_model=Decision)
async def endpoint_function(data: Query) -> Decision:
    sw = AsyncSwitch(
        states=data.states,
        comment=data.comment,
        method=data.method,
        model=data.model,
    )
    if data.vote:
        response, *_ = await vote(data.query, sw, 3)
    else:
        response, *_ = await sw.get_state(data.query)
    return {"response": response}
