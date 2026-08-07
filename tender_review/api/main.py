"""ASGI import target: ``uvicorn tender_review.api.main:app``."""

from tender_review.bootstrap import create_api_app

app = create_api_app()
