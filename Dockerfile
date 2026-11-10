# syntax=docker/dockerfile:1

# Build stage: resolve the locked dependencies into a virtual environment.
FROM python:3.13-slim AS build
RUN pip install --no-cache-dir uv==0.11.32
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never
# Dependencies first, so editing the source does not reinstall them.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --all-extras --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev --all-extras --no-editable

# Runtime stage: the virtual environment and the example configuration, no build tools.
FROM python:3.13-slim
RUN useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin dgate
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY examples ./examples
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DGATE_QUESTIONS_FILE=/app/examples/questions.yaml \
    DGATE_ROUTING_FILE=/app/examples/routing.yaml
USER dgate
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2)"]
CMD ["dgate", "serve", "--host", "0.0.0.0", "--port", "8080"]
