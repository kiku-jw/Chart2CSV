<div align="center">
  
# Chart2CSV

### Zero-Click AI Chart Data Extraction

**Extract data from chart images to CSV. No clicking required.**

[Live Demo](https://kiku-jw.github.io/Chart2CSV/) · [Documentation](https://github.com/kiku-jw/Chart2CSV/wiki) · [Report Bug](https://github.com/kiku-jw/Chart2CSV/issues)

[![Demo](https://img.shields.io/badge/demo-live-00ff88?style=for-the-badge)](https://kiku-jw.github.io/Chart2CSV/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-3776ab?style=for-the-badge)](https://python.org)

---

</div>

## ⚡ The Problem

You have a chart image from a research paper, report, or website. You need the actual numbers.

**Traditional tools** make you click each data point manually. That takes 5-30 minutes per chart.

**Chart2CSV** uses AI to read the chart automatically. Drop image → Get CSV. Done in seconds.

---

## 🚀 Quick Start

```bash
pip install chart2csv
```

```bash
python -m chart2csv.cli.main your_chart.png
```

That's it. Check `your_chart.csv` for the extracted data.

---

## 🌐 Try Online

**[Open Live Demo →](https://kiku-jw.github.io/Chart2CSV/)**

No installation needed. Works in your browser.

---

## ✨ Features

| | Feature | Description |
|---|---|---|
| ⚡ | **Zero-Click** | AI understands your chart automatically |
| 🧠 | **Smart OCR** | Mistral Vision reads axis labels accurately |
| 🔒 | **Privacy** | Runs offline by default with Tesseract |
| 📊 | **Multi-Chart** | Line, scatter, bar charts supported |
| ⚙️ | **CLI** | Batch process folders of charts |
| ✓ | **Overlay** | Visual verification of detected points |

---

## 📊 Comparison

| | WebPlotDigitizer | PlotDigitizer Pro | **Chart2CSV** |
|---|---|---|---|
| **Method** | Manual clicking | Semi-auto | AI automatic |
| **Speed** | 5-30 min | 2-10 min | **Seconds** |
| **Price** | Free | Paid | **Free** |
| **Offline** | ✓ | ✗ | ✓ |
| **CLI/API** | ✗ | ✗ | ✓ |
| **AI OCR** | ✗ | ✗ | ✓ |

---

## 💻 Usage

### Basic
```bash
python -m chart2csv.cli.main chart.png
```

### With AI (better accuracy)
```bash
export MISTRAL_API_KEY=your_key
python -m chart2csv.cli.main chart.png --use-mistral
```

### Batch processing
```bash
python -m chart2csv.cli.main charts/ --batch --output-dir results/
```

### Visual verification
```bash
python -m chart2csv.cli.main chart.png --overlay check.png
```

---

## 📖 Documentation

- [Installation](https://github.com/kiku-jw/Chart2CSV/wiki/Installation)
- [Quick Start](https://github.com/kiku-jw/Chart2CSV/wiki/Quick-Start)
- [CLI Reference](https://github.com/kiku-jw/Chart2CSV/wiki/CLI-Reference)
- [How It Works](https://github.com/kiku-jw/Chart2CSV/wiki/How-It-Works)
- [FAQ](https://github.com/kiku-jw/Chart2CSV/wiki/FAQ)

---

## 🛠️ Installation

```bash
pip install chart2csv
```

For offline OCR:
```bash
# macOS
brew install tesseract

# Ubuntu
apt-get install tesseract-ocr
```

---

## 📄 License

MIT License. Free for personal and commercial use.

---

<div align="center">

**[⬆ Back to Top](#chart2csv)**

Made with ❤️ by [kiku-jw](https://github.com/kiku-jw)

</div>
