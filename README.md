# Chart2CSV

**Extract data from chart images to CSV. No clicking required.**

[![Demo](https://img.shields.io/badge/Try-Live_Demo-brightgreen)](https://kiku-jw.github.io/Chart2CSV/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

## What is Chart2CSV?

Chart2CSV is a tool that reads chart images and extracts the data as CSV.

**Input:** A picture of a chart (PNG, JPG)  
**Output:** CSV file with x,y coordinates

Works with line charts, scatter plots, and bar charts.

## Quick Start

```bash
# Install
pip install chart2csv

# Extract data from a chart
python -m chart2csv.cli.main chart.png

# Output: chart.csv with the extracted data
```

## Live Demo

**[Try it in your browser →](https://kiku-jw.github.io/Chart2CSV/)**

No installation needed. Uses AI to read your chart.

## Why Chart2CSV?

| Tool | How it works | Speed |
|------|--------------|-------|
| WebPlotDigitizer | Click each point manually | Slow |
| PlotDigitizer Pro | Semi-automatic, paid | Medium |
| **Chart2CSV** | AI reads automatically | **Fast** |

## Features

- **Zero-click extraction** — AI understands your chart
- **Works offline** — Uses local OCR by default
- **Mistral AI option** — Better accuracy with `--use-mistral`
- **Batch processing** — Process folders of charts
- **Visual verification** — See what was detected with `--overlay`

## Usage Examples

```bash
# Basic extraction
python -m chart2csv.cli.main chart.png

# Use AI for better accuracy
python -m chart2csv.cli.main chart.png --use-mistral

# Process multiple charts
python -m chart2csv.cli.main folder/ --batch --output-dir results/

# Verify with overlay
python -m chart2csv.cli.main chart.png --overlay check.png
```

## Supported Charts

- ✅ Line charts
- ✅ Scatter plots
- ✅ Bar charts
- ✅ Linear and log scales

## Installation

```bash
pip install chart2csv
```

For offline OCR, also install Tesseract:
- macOS: `brew install tesseract`
- Ubuntu: `apt-get install tesseract-ocr`
- Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

## Documentation

- [Quick Start](https://github.com/kiku-jw/Chart2CSV/wiki/Quick-Start)
- [CLI Reference](https://github.com/kiku-jw/Chart2CSV/wiki/CLI-Reference)
- [FAQ](https://github.com/kiku-jw/Chart2CSV/wiki/FAQ)
- [Troubleshooting](https://github.com/kiku-jw/Chart2CSV/wiki/Troubleshooting)

## How It Works

1. Detects the chart area
2. Finds the X and Y axes
3. Reads the axis labels with OCR
4. Extracts data points
5. Outputs CSV

## Contributing

Issues and pull requests welcome. See [GitHub Issues](https://github.com/kiku-jw/Chart2CSV/issues).

## License

MIT License. Free for personal and commercial use.

---

**Keywords:** chart digitizer, extract data from graph, plot to csv, graph data extraction, chart image to data, webplotdigitizer alternative, digitize chart, extract values from chart image, chart ocr, graph to excel
