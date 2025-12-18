# How It Works

Chart2CSV uses computer vision and OCR to extract data from chart images.

## Pipeline Overview

```
Image → Preprocess → Detect Axes → OCR Labels → Extract Points → Output CSV
```

## Step by Step

### 1. Image Preprocessing

- Convert to grayscale
- Enhance contrast
- Remove noise

### 2. Plot Area Detection

- Find the chart boundaries
- Crop to the actual plot region
- Ignore titles, legends, and margins

### 3. Axis Detection

- Find X and Y axis lines using line detection
- Determine axis positions in pixels

### 4. Tick Label OCR

Two backends available:

**Tesseract (offline):**
- Free, runs locally
- Good for clear, high-resolution images

**Mistral Vision (cloud):**
- Uses AI to read text
- Better accuracy, especially for difficult fonts
- Requires API key

### 5. Coordinate Transformation

- Map pixel positions to actual values
- Uses detected tick labels to build the mapping
- Handles linear and logarithmic scales

### 6. Data Point Extraction

Depends on chart type:

**Scatter plots:** Detect colored dots using blob detection

**Line charts:** Trace the line using skeletonization

**Bar charts:** Detect rectangles and measure heights

### 7. Output

- CSV file with x,y coordinates
- JSON metadata (confidence, warnings, parameters)
- Visual overlay showing detected points

## Confidence Scoring

Chart2CSV reports a confidence score (0.0 to 1.0):

| Score | Meaning | Action |
|-------|---------|--------|
| ≥ 0.7 | High confidence | Trust the results |
| 0.4 - 0.7 | Medium | Check the overlay |
| < 0.4 | Low | Use manual calibration |

## Caching

OCR results are cached to speed up repeated extractions:
- Cache location: `~/.cache/chart2csv/ocr/`
- Separate caches for Tesseract and Mistral
- Use `--no-cache` to bypass
