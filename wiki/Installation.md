# Installation

## Requirements

- Python 3.8 or newer
- Tesseract OCR (for offline mode)

## Install Chart2CSV

```bash
pip install chart2csv
```

Or install from source:

```bash
git clone https://github.com/kiku-jw/Chart2CSV.git
cd Chart2CSV
pip install -e .
```

## Install Tesseract OCR

Tesseract is needed for offline text recognition.

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## Optional: Mistral API Key

For better accuracy, you can use Mistral AI instead of Tesseract.

1. Get a free API key at https://console.mistral.ai/
2. Set it as environment variable:

```bash
export MISTRAL_API_KEY=your_key_here
```

3. Use `--use-mistral` flag when running Chart2CSV

## Verify Installation

```bash
python -m chart2csv.cli.main --help
```

You should see the help message with all available options.
