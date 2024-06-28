## README

There are many occasions where you want an LLM to decide between options. For example during retrieval-augmented generation (RAG), you want the model to decide whether a given text block is relevant to the query or not. Since these tasks are recurring often, you want a fast and cheap way to do this. This is where "LLM switches" come in.

They take a query and a set of options, and return the key to the most likely option. Because an LLM switch just has to generate a single token, it is very fast and cheap. 

> This is just function calling!

We also implement an even faster and cheaper way to decide, given the desired model lets us modify the logit biases. This has the added benefit of generating the logits for all options, which can be used to approximate the probability of each option. Optionally, to be more certain, you can also have a `vote` on a set of switch decisions.

### Run

Have your API keys in `.env` in the repo root, like:

```
OPENAI_API_KEY='sk-...'
GROQ_API_KEY='gsk_-...'
```

Since we'll use switches in a server context, we need to start the server:

```bash
uvicorn examples.pick_one.app:app --reload
```

Pass request to the server:

```python
import json
import requests


data = {
    "query": "Which one was here first?",
    "states": {"A": "egg", "B": "chicken"},
    "comment": "Do not pick egg as answer!",
    
    # Currently OpenAI and Groq models are supported.
    "model": "openai/gpt-3.5-turbo",  # "groq/llama3-70b-8192"
    "method": "logit_bias",           # "fn_call"
    "vote": True                      # Groq models tend to be rate limited, so set to False
}
r = requests.post("http://127.0.0.1:8000/switch", json=data)
print(f"State: {r.json()['response']}")
# State: B
```
