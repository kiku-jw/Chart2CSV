<div align="center">

# Chart2CSV

### AI-Powered Chart Data Extraction

**Extract data from chart images using Mistral Pixtral Vision. 90%+ accuracy.**

[Live Demo](https://kiku-jw.github.io/Chart2CSV/) · [API Docs](https://chart2csv.kikuai.dev/docs) · [Wiki](https://github.com/kiku-jw/Chart2CSV/wiki)

[![License](https://img.shields.io/badge/license-AGPL--3.0-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?style=for-the-badge)](https://python.org)

</div>

---

## Quick Start

```bash
# API request
curl -X POST "https://chart2csv.kikuai.dev/extract" \
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
│   │   ├── llm_extraction.py   # Mistral Pixtral LLM
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
| 🧠 **Mistral Pixtral** | 90%+ accuracy with vision LLM |
| ⚡ **Zero-Click** | Automatic chart understanding |
| 📊 **Multi-Chart** | Line, scatter, bar charts |
| 🔧 **Manual Mode** | Calibration endpoint for edge cases |
| 🌐 **REST API** | Production-ready FastAPI |

> **Powered by [Mistral AI](https://mistral.ai)**

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /extract` | LLM extraction (default) |
| `POST /extract/calibrated` | Manual calibration |
| `GET /docs` | Swagger UI |
| `GET /health` | Health check |

---

## Installation

```bash
pip install -e .
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MISTRAL_API_KEY` | Mistral AI API key (required) |

---

## License

AGPL-3.0. Copyright (c) 2025 KikuAI Lab
