## API

Request:

- Not sending a `session_id` will create a new session, or reset the current session.

```json
{
    "token": "string",
    "user_id": "string",
    "session_id": "string" | null,
    "query": "string" | null
}
```

Response:

- The response can contain multiple messages, for example when the first provides (generated) feedback on the answer and the second poses the next question.

```json
[
    {
        "user_id": "string",
        "session_id": "string",
        "response": {
            "text": "string",
            "hint": "string" | null,
            "evaluation": "string" | null,
            "image": "string" | null
        }
    }
]
```

The app receives a hint, which it can display if the user wishes.
