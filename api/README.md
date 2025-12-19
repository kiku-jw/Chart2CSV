# API

FastAPI REST API for Chart2CSV.

## Files

| File | Description |
|------|-------------|
| `main.py` | API endpoints: `/extract`, `/extract/calibrated`, `/health` |
| `__init__.py` | Package init |

## Endpoints

- `POST /extract` — LLM extraction (default)
- `POST /extract/calibrated` — Manual calibration mode
- `POST /extract/base64` — Base64 image input
- `GET /health` — Health check
- `GET /docs` — Swagger UI

## Running

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
