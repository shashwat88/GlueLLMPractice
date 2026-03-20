FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy packaging metadata first for better layer caching.
COPY pyproject.toml README.md ./
COPY agents ./agents
COPY core ./core
COPY tools ./tools

RUN pip install --no-cache-dir .

# Run as non-root in the final container.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Default neutral command; override per run via docker compose run.
CMD ["python", "--version"]
