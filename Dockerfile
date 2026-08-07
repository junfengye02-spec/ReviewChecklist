FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system tender-review \
    && useradd --system --gid tender-review --home-dir /app tender-review

COPY requirements.txt ./
RUN python -m pip install --requirement requirements.txt

COPY . ./
RUN chown -R tender-review:tender-review /app

USER tender-review

EXPOSE 8000

CMD ["uvicorn", "tender_review.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
