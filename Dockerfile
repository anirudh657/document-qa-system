FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir hatchling
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.11-slim

RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels document-qa-system \
    && rm -rf /wheels

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/v1/health').status_code==200 else 1)"

CMD ["uvicorn", "rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
