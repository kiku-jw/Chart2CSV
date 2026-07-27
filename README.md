<div align="center">

# Chart2CSV

Chart2CSV turns chart images into CSV-style data for analysts and operators who need a quick extraction path from visual reports.

**Status: Alpha. Verify extracted values against the source chart before using them.**

**[Try the live demo](https://kikuai-lab.github.io/Chart2CSV/)**

[Docs](https://github.com/KikuAI-Lab/Chart2CSV/wiki) · [Quick start](#quick-start) · [API request](#quick-start)

Illustrative CSV format (not measured output):

```csv
label,value
Q1,42
Q2,57
```

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?style=for-the-badge)](https://python.org)

</div>

---

## Quick Start

```bash
# In one terminal, set MISTRAL_API_KEY in the environment and start the local API:
python -m uvicorn api.main:app --reload

# In another terminal:
curl -X POST "http://127.0.0.1:8000/v1/extract" \
  -F "file=@chart.png"
```

---

## Project Structure

```
Chart2CSV/
├── api/                    # FastAPI REST API
│   └── main.py             # API endpoints
├── chart2csv/              # Core Python package
│   ├── core/               # Extraction logic
│   │   ├── llm_extraction.py   # Mistral-backed extraction
│   │   ├── pipeline.py         # CV pipeline (fallback)
│   │   └── ocr.py              # OCR for axis labels
│   └── cli/                # Command-line interface
├── deploy/                 # Deployment scripts
│   ├── deploy.sh           # Server deployment
│   └── nginx.conf          # Nginx config
├── scripts/                # Development utilities
├── Dockerfile              # Container build
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Python dependencies
└── setup.py                # Package installation
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Mistral extraction** | Vision-model extraction path |
| ⚡ **Automatic extraction** | Reads supported charts without manual point selection |
| 📊 **Multi-Chart** | Line, scatter, bar charts |
| 🔧 **Manual Mode** | Calibration endpoint for edge cases |
| 🌐 **REST API** | FastAPI endpoints for extraction workflows |

> **Powered by [Mistral AI](https://mistral.ai)**

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/extract` | LLM extraction (default) |
| `POST /v1/extract/calibrated` | Manual calibration |
| `GET /docs` | Swagger UI |
| `GET /health` | Health check |

---

## Data Handling

The browser demo sends the selected chart image and user-provided API key directly to Mistral. It stores the API key in that browser's `localStorage` until the field or site data is cleared.

The local API uses `MISTRAL_API_KEY` for its default LLM-backed extraction path. The CLI uses the local CV/Tesseract path unless `--use-mistral` is supplied. Mistral-backed modes transmit image data to Mistral.

---

## Installation

```bash
pip install -e .
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRAL_API_KEY` | Mistral AI API key (required for Mistral-backed extraction) |

---

## License

AGPL-3.0. Copyright (c) 2025 KikuAI OÜ

<!-- author-links:start -->
<p align="center">
  <a href="https://kikuai.dev/"><img src="https://img.shields.io/badge/Website-kikuai.dev-111827?style=for-the-badge&logo=safari&logoColor=white" alt="KikuAI website"></a>
  <a href="https://t.me/kiku_ai"><img src="https://img.shields.io/badge/Telegram-%40kiku__ai-26A5E4?style=for-the-badge&logo=telegram&logoColor=white" alt="Telegram @kiku_ai"></a>
  <a href="https://github.com/kiku-jw"><img src="https://img.shields.io/badge/GitHub-%40kiku--jw-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub @kiku-jw"></a>
</p>
<p align="center">
  <sub>Follow new projects and updates from <a href="https://github.com/kiku-jw">@kiku-jw</a>.</sub>
</p>
<!-- author-links:end -->
