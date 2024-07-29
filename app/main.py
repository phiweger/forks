from contextlib import asynccontextmanager
from pathlib import Path
import tomllib as toml
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from jinja2_fragments.fastapi import Jinja2Blocks
from loguru import logger
from sqlmodel import create_engine, SQLModel

from forks.graph import load_graph, GraphTraversal
from forks.models import Query

load_dotenv()
shared = {}
with open("config.toml", "rb") as file:
    config = toml.load(file)
BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Blocks(directory="templates")
# Vanilla jinja:
# TEMPLATES = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load shared information on startup -- https://fastapi.tiangolo.com/advanced/events/
    """

    shared["graphs"] = {}

    p = Path("data/graphs/")
    for file in p.glob("*.toml"):
        name = file.stem
        with open(f"data/graphs/{name}.toml", "rb") as file:
            data = toml.load(file)
            G = load_graph(data)
            G.name = name
        shared["graphs"][name] = G

    if not (db := os.environ["DATABASE_URI"]):
        logger.warning("No database URI found, using default.")
        db = "sqlite:///tmp/test.db"
    engine = create_engine(db, echo=True, pool_pre_ping=True)
    shared["engine"] = engine

    def create_db_and_tables(engine):
        SQLModel.metadata.create_all(engine)

    create_db_and_tables(engine)

    yield

    # Clean up the ML models and release the resources
    shared.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", status_code=200)
async def root(request: Request) -> dict:
    # return {"Hello": "World"}
    return TEMPLATES.TemplateResponse(
        "index.html",
        {"request": request, "graphs": shared["graphs"].keys()},
    )


@app.post("/route", status_code=200)  # response_class=HTMLResponse
async def get_next_step(payload: Request) -> dict:

    params = config["route"]

    # payload.headers
    # Sanitize input:
    # https://github.com/pydantic/pydantic/discussions/4597
    data = Query.model_validate(await payload.json())

    try:
        G = shared["graphs"][data.graph]
    except KeyError:
        msg = f"Graph {data.graph} not found."
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg)

    try:
        trav = GraphTraversal(G, shared["engine"], data.user_id, data.session_id)
    except Exception as e:
        # Bubble up the error message.
        msg = str(e)
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    next_step = trav.get_next_node()
    if trav.is_leaf(next_step):
        msg = "The current node is a leaf. Reset the session to continue."
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    # If there is no query, we just return the current node.
    if data.query:
        current = next_step
        next_step = await trav.route(node=current, query=data.query, **params)
        trav.log_state(current, next_step, data.query)

    response = {
        "session_id": trav.session_id,
        "content": trav.get_node_content(next_step),
        "end": trav.is_leaf(next_step),
    }

    logger.info(response)
    return response
