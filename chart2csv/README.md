# chart2csv

Core Python package for chart data extraction.

## Structure

```
chart2csv/
├── __init__.py      # Package init, version
├── cli/             # Command-line interface
│   └── main.py      # CLI entry point
├── core/            # Core extraction logic
│   ├── llm_extraction.py   # Mistral Pixtral LLM extraction
│   ├── pipeline.py         # CV extraction pipeline
│   ├── ocr.py              # OCR for axis labels
│   ├── mistral_ocr.py      # Mistral OCR backend
│   ├── extraction.py       # Point extraction
│   ├── detection.py        # Chart/axis detection
│   └── types.py            # Data types
└── tests/           # Unit tests
```

## Usage

```python
from chart2csv.core.llm_extraction import extract_chart_llm

result, confidence = extract_chart_llm("chart.png")
print(result["data"])  # [{x: 0, y: 10}, ...]
```
