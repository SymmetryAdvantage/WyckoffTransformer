### updated from https://depot.dev/docs/container-builds/optimal-dockerfiles/python-poetry-dockerfile

# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
ENV POETRY_VERSION=2.2.1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"
RUN pip install "poetry==$POETRY_VERSION"
WORKDIR $PYSETUP_PATH
COPY docker/poetry.lock docker/pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pypoetry \
    poetry lock && poetry install --no-root
FROM python:3.12-slim AS runtime
RUN apt update && apt install -y vim build-essential
ENV VENV_PATH="/opt/pysetup/.venv" \
    PATH="/opt/pysetup/.venv/bin:$PATH"
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -d /home/appuser -s /bin/bash appuser
WORKDIR /home/appuser
COPY --from=build --chown=appuser:appgroup /opt/pysetup/.venv /opt/pysetup/.venv
COPY --from=build --chown=appuser:appgroup /opt/pysetup/poetry.lock /opt/pysetup/pyproject.toml /opt/wyformer_app/
COPY --chown=appuser:appgroup . /opt/wyformer_app/WyckoffTransformer
USER appuser
RUN mkdir /home/appuser/workdir
WORKDIR /home/appuser/workdir

# prepare pilot model:
RUN wandb offline
RUN python /opt/wyformer_app/WyckoffTransformer/scripts/preprocess_wychoffs.py && \
    python /opt/wyformer_app/WyckoffTransformer/scripts/cache_a_dataset.py mp_20 && \
    python /opt/wyformer_app/WyckoffTransformer/scripts/tokenise_a_dataset.py mp_20 \
        /opt/wyformer_app/WyckoffTransformer/yamls/tokenisers/mp_20_sg_multiplicity.yaml --new-tokenizer
