# TODO
# use case: agentic traversal of a decision tree


import networkx as nx


# Interface
"""
paths = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
}

annotation = {
    'A': {
        'name': 'A',
        'type': 'start',
    },
}

G = nx.Graph(paths)
new = G.get_next_step(current_node, response)
# Uses switches internally.
"""
