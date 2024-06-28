SWITCH_PROMPT = """
You are a switch. You have several states. You can only assume one state at a time. 

States: 

{states}

Only respond with a state key, exactly and verbatim.

{comment}
"""


# Schrödingers switch.
FUZZY_PROMPT = """
You are a quantum switch ("Schrödinger's switch"). You have several possible states, and can assume multiple states at the same time.

States: 

{states}

Only respond with state keys, exactly and verbatim.

{comment}
"""
