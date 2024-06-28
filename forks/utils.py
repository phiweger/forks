import os
from typing import Tuple

from dotenv import load_dotenv
from groq import Groq, AsyncGroq
import instructor
from openai import OpenAI, AsyncOpenAI


load_dotenv()


# TODO:
# Anthropic
# https://github.com/jxnl/instructor/blob/main/examples/anthropic/run.py


def get_groq_client(use_async: bool = True, structured: bool = True) -> Tuple[AsyncGroq, Groq]:  # type: ignore
    # https://python.useinstructor.com/examples/groq/#use-example
    key = os.environ.get("GROQ_API_KEY")
    if use_async:
        client = AsyncGroq(api_key=key)
    else:
        client = Groq(api_key=key)

    if not structured:
        return client
    else:
        return instructor.from_groq(client, mode=instructor.Mode.TOOLS)


def get_openai_client(use_async: bool = True, structured: bool = True) -> Tuple[AsyncOpenAI, OpenAI]:  # type: ignore
    key = os.environ.get("OPENAI_API_KEY")
    if use_async:
        client = AsyncOpenAI(api_key=key)
    else:
        client = OpenAI(api_key=key)

    if not structured:
        return client
    else:
        return instructor.patch(client)


def get_client(provider: str = "openai", use_async: bool = True, structured: bool = True) -> Tuple[Groq, OpenAI]:  # type: ignore
    allowed = [
        "openai",
        "groq",
    ]
    # The provider can be read from the typical hugging face style of model specification:
    # "openai/gpt-3.5-turbo" > "openai"
    if "/" in provider:
        provider, model = provider.split("/")
    else:
        model = None

    # Pick a client.
    if provider == "openai":
        client = get_openai_client(use_async=use_async, structured=structured)
    elif provider == "groq":
        client = get_groq_client(use_async=use_async, structured=structured)
    else:
        raise ValueError("Invalid provider, choose from: " + ", ".join(allowed))

    return client, model
