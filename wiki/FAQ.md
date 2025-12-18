# Frequently Asked Questions

Common questions about Chart2CSV.

---

## What is Chart2CSV?

Chart2CSV is a tool that extracts numerical data from chart images. You give it a picture of a chart (like a screenshot or scan), and it gives you the data as a CSV file.

---

## How do I extract data from a chart image?

```bash
python -m chart2csv.cli.main your_chart.png
```

This creates a CSV file with the extracted data points.

---

## What chart types are supported?

- Line charts
- Scatter plots
- Bar charts

Both linear and logarithmic scales are supported.

---

## Is Chart2CSV free?

Yes. Chart2CSV is free and open source under the MIT license.

---

## Does it work offline?

Yes. By default, Chart2CSV uses Tesseract OCR which runs completely offline. Your images never leave your computer.

---

## What is Mistral mode?

Mistral is an AI service that can read text from images more accurately than traditional OCR. To use it:

1. Get a free API key from https://console.mistral.ai/
2. Set `MISTRAL_API_KEY` environment variable
3. Add `--use-mistral` flag

---

## How accurate is it?

Accuracy depends on image quality. For clean, high-resolution charts, accuracy is typically 95%+. 

Use `--overlay` to verify results visually.

---

## What if the results are wrong?

1. Check the overlay image to see what was detected
2. Try `--use-mistral` for better OCR
3. Use `--crop` to focus on the chart area
4. Use `--calibrate` for manual calibration

---

## Can I process multiple charts at once?

Yes. Use batch mode:

```bash
python -m chart2csv.cli.main folder/ --batch --output-dir results/
```

---

## What image formats are supported?

- PNG
- JPG/JPEG
- WebP

---

## How do I cite Chart2CSV?

```bibtex
@software{chart2csv,
  title = {Chart2CSV: Zero-Click AI Chart Data Extraction},
  author = {kiku-jw},
  year = {2025},
  url = {https://github.com/kiku-jw/Chart2CSV}
}
```

---

## Where can I report bugs?

Open an issue on GitHub: https://github.com/kiku-jw/Chart2CSV/issues
