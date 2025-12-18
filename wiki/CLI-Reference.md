# CLI Reference

All command line options for Chart2CSV.

## Basic Usage

```bash
python -m chart2csv.cli.main INPUT [OPTIONS]
```

## Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Path to chart image (PNG, JPG) or folder for batch mode |

## Options

### Output Options

| Option | Description | Example |
|--------|-------------|---------|
| `-o, --output FILE` | Output CSV path | `--output data.csv` |
| `--metadata FILE` | Output JSON metadata | `--metadata meta.json` |
| `--overlay FILE` | Save visual overlay | `--overlay check.png` |
| `--output-dir DIR` | Output folder (batch mode) | `--output-dir results/` |

### Chart Options

| Option | Description | Example |
|--------|-------------|---------|
| `--chart-type TYPE` | Force chart type: `scatter`, `line`, `bar` | `--chart-type scatter` |
| `--x-scale SCALE` | X-axis scale: `linear`, `log` | `--x-scale log` |
| `--y-scale SCALE` | Y-axis scale: `linear`, `log` | `--y-scale log` |

### Manual Overrides

| Option | Description | Example |
|--------|-------------|---------|
| `--crop X1,Y1,X2,Y2` | Crop to specific region | `--crop 50,30,750,600` |
| `--x-axis Y` | X-axis Y position (pixels) | `--x-axis 550` |
| `--y-axis X` | Y-axis X position (pixels) | `--y-axis 80` |
| `--calibrate` | Manual calibration mode | `--calibrate` |

### OCR Options

| Option | Description | Example |
|--------|-------------|---------|
| `--use-mistral` | Use Mistral AI for OCR | `--use-mistral` |
| `--no-cache` | Disable OCR result caching | `--no-cache` |

### Batch Options

| Option | Description | Example |
|--------|-------------|---------|
| `--batch` | Process folder of images | `--batch` |

## Examples

### Basic extraction
```bash
python -m chart2csv.cli.main chart.png
```

### With overlay verification
```bash
python -m chart2csv.cli.main chart.png --overlay check.png
```

### Using Mistral AI
```bash
python -m chart2csv.cli.main chart.png --use-mistral
```

### Batch processing
```bash
python -m chart2csv.cli.main figures/ --batch --output-dir results/
```

### Manual crop
```bash
python -m chart2csv.cli.main chart.png --crop 100,50,800,600
```

### Log scale chart
```bash
python -m chart2csv.cli.main chart.png --y-scale log
```
