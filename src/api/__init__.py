"""FastAPI service exposing the latent dynamics engine over HTTP.

All endpoints carry the non-clinical research disclaimer in every response.
This package is import-safe even when FastAPI is not installed; the actual
app object is constructed lazily inside ``main.py``.
"""
