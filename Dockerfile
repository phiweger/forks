FROM python:3.11-bookworm as builder

RUN apt-get -y clean && apt-get -y update && apt-get -y install libpq-dev gcc python3-psycopg2
RUN pip install poetry==1.8.3
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN touch README.md
RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --without dev --no-root

FROM python:3.11-bookworm as runtime

RUN apt-get -y clean && apt-get -y update && apt-get -y install libpq-dev gcc python3-psycopg2
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ADD . .
RUN useradd -m myuser
USER myuser
# https://github.com/heroku/alpinehelloworld/blob/master/Dockerfile#L24C40-L24C40
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
