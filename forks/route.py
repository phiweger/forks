from forks.switch import AsyncSwitch


def get_options(G, current_state):
    states = {}
    for child in G.neighbors(current_state):
        edge = (current_state, child)
        attrs = G.edges[edge]
        states[edge] = attrs["if"]
        comment = attrs.get("comment")
    return states, comment if comment else ""


async def route_query(graph, state, query, **kwargs):
    states, comment = get_options(graph, state)
    sw = AsyncSwitch(states=states, comment=comment, **kwargs)
    next_step, *_ = await sw.get_state(query)
    return next_step
