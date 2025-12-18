# Troubleshooting

Solutions to common problems.

---

## "No module named 'cv2'"

**Problem:** OpenCV is not installed.

**Solution:**
```bash
pip install opencv-python
```

---

## "pytesseract.TesseractNotFoundError"

**Problem:** Tesseract OCR is not installed.

**Solution:**

macOS:
```bash
brew install tesseract
```

Ubuntu:
```bash
sudo apt-get install tesseract-ocr
```

Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

---

## Low confidence score

**Problem:** Chart2CSV reports confidence < 0.5

**Solutions:**
1. Use higher resolution image
2. Crop to just the chart area: `--crop x1,y1,x2,y2`
3. Try Mistral OCR: `--use-mistral`
4. Use manual calibration: `--calibrate`

---

## Wrong axis values

**Problem:** Extracted values don't match the chart.

**Solutions:**
1. Check if scale is logarithmic: `--y-scale log`
2. Use manual calibration: `--calibrate`
3. Verify with overlay: `--overlay check.png`

---

## Too many or too few points detected

**Problem:** Point count doesn't match visual.

**Solutions:**
1. Force chart type: `--chart-type scatter`
2. Adjust crop to exclude legend/title: `--crop`
3. Check overlay to see what was detected

---

## Mistral API errors

**Problem:** "API request failed" or authentication error.

**Solutions:**
1. Check API key is set: `echo $MISTRAL_API_KEY`
2. Verify key is valid at https://console.mistral.ai/
3. Check internet connection
4. Fall back to Tesseract (remove `--use-mistral`)

---

## Slow processing

**Problem:** Takes too long to process.

**Solutions:**
1. OCR results are cached - second run is faster
2. Reduce image size before processing
3. Use Tesseract instead of Mistral for speed

---

## Still stuck?

Open an issue with:
1. Your command
2. Error message
3. Sample image (if possible)

https://github.com/kiku-jw/Chart2CSV/issues
