# Chart2CSV API Reference

> **Base URL:** `https://chart2csv.kikuai.dev`
> 
> **Powered by Mistral Pixtral** — 90%+ accuracy on chart extraction

---

## Endpoints

### `POST /extract`

Extract data from a chart image.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Chart image (PNG, JPG, WebP) |
| `mode` | string | `llm` | Extraction mode: `llm`, `cv`, `auto` |
| `chart_type` | string | auto | Force type: `scatter`, `line`, `bar` |

#### Modes

- **`llm`** (default) — Mistral Pixtral vision extraction, 90%+ accuracy
- **`cv`** — Computer vision pipeline (legacy), faster but less accurate
- **`auto`** — Try LLM first, fall back to CV if it fails

---

## Examples

### curl

```bash
curl -X POST "https://chart2csv.kikuai.dev/extract" \
  -F "file=@chart.png" \
  -F "mode=llm"
```

### Python

```python
import requests

url = "https://chart2csv.kikuai.dev/extract"

with open("chart.png", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={"mode": "llm"}
    )

result = response.json()
print(f"Extracted {len(result['data'])} points")
print(f"CSV:\n{result['csv']}")
```

### JavaScript (fetch)

```javascript
const form = new FormData();
form.append('file', fileInput.files[0]);
form.append('mode', 'llm');

const response = await fetch('https://chart2csv.kikuai.dev/extract', {
  method: 'POST',
  body: form
});

const result = await response.json();
console.log(`Extracted ${result.data.length} points`);
console.log('CSV:', result.csv);
```

### JavaScript (axios)

```javascript
import axios from 'axios';

const form = new FormData();
form.append('file', file);
form.append('mode', 'llm');

const { data } = await axios.post(
  'https://chart2csv.kikuai.dev/extract',
  form
);

console.log(data.csv);
```

---

## Response Format

```json
{
  "success": true,
  "chart_type": "scatter",
  "confidence": 0.95,
  "data": [
    {"x": 0, "y": 10},
    {"x": 1, "y": 20},
    {"x": 2, "y": 30}
  ],
  "csv": "x,y\n0,10\n1,20\n2,30",
  "warnings": [],
  "processing_time_ms": 2500
}
```

---

## Rate Limits

- **20 requests per minute** per IP
- Contact us for higher limits

---

## Swagger UI

Interactive API documentation: [chart2csv.kikuai.dev/docs](https://chart2csv.kikuai.dev/docs)
