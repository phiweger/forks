import itertools
from uuid import uuid4

from loguru import logger
import networkx as nx
from sqlmodel import select, Session

from forks.db import GraphState
from forks.route import route_query


def load_graph(data):
    """Parse TOML data into a networkx graph."""
    G = nx.DiGraph()
    for node, v in data.items():
        try:
            G.add_edges_from(
                [(node, child, attrs) for child, attrs in v.get("to").items()]
            )
        except AttributeError:
            # No children, leaf node.
            # 'NoneType' object has no attribute 'keys'
            pass
        for field, attrs in v.items():
            if not field == "to":
                G.nodes[node][field] = attrs
    return G


def create_new_graph_session():
    logger.info("Creating new session.")
    return uuid4().__str__()


def get_node_content(graph, node_name: str):
    return graph.nodes[node_name]


def get_travel_state(
    location="next",
    engine=None,
    graph=None,
    user_id: str = None,
    session_id: str = None,
):
    if not graph or not user_id:
        raise ValueError("Need graph and user ID.")

    if not engine:
        raise ValueError("Need database engine.")

    with Session(engine) as session:
        statement = (
            select(GraphState)
            .where(
                GraphState.user_id == user_id,
                GraphState.session_id == session_id,
            )
            .order_by(GraphState.timestamp.desc())
        )
        # Get the most recent state.
        state = session.exec(statement).first()
        if not state:
            raise ValueError("No records found. Check user and session ID.")
            # We could just create a new session here, but rather fail here bc/ we can assume some error upstream.

    response = state.route_to if location == "next" else state.current_node
    return response


def get_last_session(engine, user_id):
    with Session(engine) as session:
        statement = (
            select(GraphState)
            .where(
                GraphState.user_id == user_id,
            )
            .order_by(GraphState.timestamp.desc())
        )
        state = session.exec(statement).first()
        return state


def sliding_window(seq, size=2):
    """
    https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, size))
    if len(result) == size:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class GraphTraversal:
    def __init__(self, graph, engine, user_id, session_id=None):
        self.graph = graph
        self.user_id = user_id
        self.engine = engine

        # Create a new session if none is provided.
        if not session_id:
            self.reset()  # sets self.session_id
        else:
            last_session = get_last_session(self.engine, self.user_id).session_id
            if not last_session == session_id:
                raise ValueError("Session ID does not match last session.")
            else:
                self.session_id = session_id
        return None

    def get_next_node(self):
        """
        Passing None as session ID will create a new session.
        """
        return get_travel_state(
            "next", self.engine, self.graph, self.user_id, self.session_id
        )

    def get_current_node(self):
        """
        Passing None as session ID will create a new session.
        """
        return get_travel_state(
            "current", self.engine, self.graph, self.user_id, self.session_id
        )

    async def route(
        self,
        node="root",
        query="",
        model="openai/gpt-4o-mini",
        method="fn_call",
    ):
        """
        log .. ability to turn off logging for testing
        """
        _, next_step = await route_query(
            self.graph, node, query, model=model, method=method
        )
        # Add a new row to the database with the new state.
        return next_step

    def reset(self):
        self.session_id = create_new_graph_session()
        self.log_state(None, "root", None)
        return True

    def get_node_content(self, node: str) -> dict:
        return get_node_content(self.graph, node)

    def is_leaf(self, node: str) -> bool:
        return self.graph.out_degree(node) == 0
        # Alternative:
        # return not G.successors(node)

    def get_last_correct_node(self):
        with Session(self.engine) as session:
            statement = (
                select(GraphState)
                .where(
                    GraphState.user_id == self.user_id,
                    GraphState.session_id == self.session_id,
                )
                .order_by(GraphState.timestamp.desc())
            )  # from most recent to oldest
            state = session.exec(statement).all()

        for target, source in sliding_window(state):
            is_correct = "correct" in self.graph.get_edge_data(
                source.route_to, target.route_to
            ).get("labels", [])
            if is_correct:
                return target.route_to
            return "root"

    def log_state(self, current_node, next_step, query):
        with Session(self.engine) as session:
            new_state = GraphState(
                graph_name=self.graph.name,
                user_id=self.user_id,
                session_id=self.session_id,
                current_node=current_node,
                response=query,
                route_to=next_step,
                comment={},
            )
            session.add(new_state)
            logger.info(f"Adding new state: {new_state}")
            session.commit()
        return True
