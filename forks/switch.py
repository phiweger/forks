from collections import defaultdict, Counter
import time
from typing import Literal

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random
import tiktoken

from forks.prompt import SWITCH_PROMPT, FUZZY_PROMPT
from forks.utils import get_client


class Tokenizer:
    """
    https://www.reddit.com/r/ChatGPTCoding/comments/126gmbh/anyone_had_success_using_logit_bias_on_gpt35/
    """

    def __init__(self, model="gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model)

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def tokenIDs_from_list(self, lst):
        tokenIDs = []
        for string in lst:
            variations = [
                string,
                string.capitalize(),
                " " + string,
                " " + string.capitalize(),
            ]
            for variation in variations:
                ids = self.encode(variation)
                for id in ids:
                    tokenIDs += [id]
        tokenIDs = list(dict.fromkeys(tokenIDs))
        return tokenIDs


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# def logprob_to_prob(logprob: list) -> list:
#     return np.exp(logprob) / np.sum(np.exp(logprob))


def get_probs(response, labels):
    """
    Can we use logprobs in softmax? Yes, I think:
    https://stats.stackexchange.com/questions/289369/log-probabilities-in-reference-to-softmax-classifier
    """
    d = {}
    for i in response.choices[0].logprobs.content[0].top_logprobs:
        if i.token in labels:
            d[i.token] = i.logprob
    if not d:
        raise ValueError("No label found in top logprobs.")

    sm = softmax(list(d.values()))
    probs = {k: round(v, 4) for k, v in zip(d.keys(), sm)}
    # return dict(sorted(probs.items()))
    return probs


def encode_labels(labels, model="gpt-3.5-turbo"):
    tk = Tokenizer(model)
    for i in labels:
        # Longer words are decomposed into more than one token.
        for j in tk.encode(i):
            yield j


def no_state(retry_state):
    logger.error(f"Retrying {retry_state}")
    return "", None, 0


# As of 2024-06-27 Groq cannot do logprobs:
# https://console.groq.com/docs/openai
# The following fields are not supported and will result in a 400 error if they are supplied:
# logprobs, logit_bias, top_logprobs
# If at some point they do, we need to make sure to use the right tokeniser:
# KeyError: 'Could not automatically map mixtral-8x7b-32768 to a tokeniser. Please use `tiktoken.get_encoding` to explicitly get the tokeniser you expect.'
# https://gist.github.com/vatsalsaglani/3f12c4213975c56a9bf1fd5cfa60f596
MODEL_CATALOGUE = {
    "openai": {
        "logit_bias": True,
        "fn_call": True,
    },
    "groq": {
        "logit_bias": False,
        "fn_call": True,
    },
}


class AsyncSwitch:
    def __init__(
        self, states, comment="", model=None, method="logit_bias", allow_many=False
    ) -> None:
        # We'll assign logprobs to tokens; thus, we need single tokens.
        # The user-provided keys could be multi-token, so we map them to numbers.
        self.monotoken = {str(new_k): v for new_k, (_, v) in enumerate(states.items())}
        self.monotoken_rev = {
            str(new_k): k for new_k, (k, _) in enumerate(states.items())
        }
        self.template = FUZZY_PROMPT if allow_many else SWITCH_PROMPT
        self.comment = comment
        self.labels = list(self.monotoken.keys())

        if method != "logit_bias" and allow_many:
            raise ValueError("Only logit bias supports multiple states for now.")
        provider = model.split("/")[0]
        assert MODEL_CATALOGUE[provider], f"Provider {provider} not supported."
        assert MODEL_CATALOGUE[provider][
            method
        ], f"Switch method '{method}' not supported for {provider}, choose among {[k for k, v in MODEL_CATALOGUE[provider].items() if v]}."

        self.method = method
        self.client, self.model = get_client(
            model, use_async=True, structured=self.method == "fn_call"
        )

        # TODO: This only works for openai models.
        if self.method == "logit_bias":
            self.biases = {
                token: 100 for token in encode_labels(self.labels, self.model)
            }
        else:
            self.biases = None

        self.instructions = self.template.format(
            states="".join(
                [f"- {k}: {v}\n" for k, v in self.monotoken.items()]
            ).strip(),
            comment=self.comment,
        )

    # reraise=True .. put retry error at end of traceback, not in the middle
    # https://gist.github.com/2minchul/da19cf0698ffa0cf32b595eecf3b463c
    @retry(wait=wait_random(min=1, max=2), stop=stop_after_attempt(3), reraise=True)
    async def get_state(self, query, temperature=0.0):
        start = time.time()

        if self.method == "logit_bias":
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                # response_model = Answer,
                logit_bias=self.biases,
                logprobs=True,
                top_logprobs=len(self.labels) + 2,  # some slack,
                max_tokens=1,
                # timeout=.5,
                # https://community.openai.com/t/recommended-way-to-limit-the-amount-of-time-a-python-chatcompletion-create-runs/373386/5
                temperature=temperature,
            )

            answer, probs = r.choices[0].message.content, get_probs(r, self.labels)

        elif self.method == "fn_call":

            class Answer(BaseModel):
                # Literal is ok as alternative to enums:
                # https://python.useinstructor.com/concepts/enums/
                # Dynamic literals:
                # https://github.com/pydantic/pydantic/issues/8905
                # answer: int = Literal[*labels]  # type: ignore
                answer: Literal[*self.labels]  # type: ignore

            r = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": f"Query: {query}"},
                ],
                response_model=Answer,
                temperature=temperature,
            )
            answer, probs = r.answer, None
            logger.info(f"Answer: {answer}")

        else:
            raise ValueError("Invalid method, available: 'logit_bias', 'fn_call'")

        # Resubstitute the labels
        answer = self.monotoken_rev[answer]
        if probs:
            probs = {self.monotoken_rev[k]: v for k, v in probs.items()}

        end = time.time()
        return answer, probs, round(end - start, 2)


def postprocess_switch_states(x):
    rr, pp, tt = zip(*x)

    # Calculate the average probabilities.
    if all(pp):
        tmp = defaultdict(list)
        for p in pp:
            for k, v in p.items():
                tmp[k].append(v)
        probs = {k: round(np.mean(v), 4) for k, v in tmp.items()}
    else:
        probs = None

    r = Counter(rr).most_common(1)[0][0]
    t = round(np.mean(tt), 2)
    return r, probs, t


# TODO
# def vote(, policy="best")  # majority
async def vote(query, switch, n_tries: int = 5):
    x = await asyncio.gather(
        *[switch.get_state(query, temperature=0.3) for _ in range(n_tries)]
    )
    return postprocess_switch_states(x)
