# Quick Start

Get your first chart extracted in 2 minutes.

## Step 1: Install

```bash
pip install chart2csv
```

## Step 2: Extract Data

Basic command:
```bash
python -m chart2csv.cli.main your_chart.png
```

This creates `your_chart.csv` with the extracted data.

## Step 3: Verify Results

Add `--overlay` to see what was detected:
```bash
python -m chart2csv.cli.main your_chart.png --overlay check.png
```

Open `check.png` to see the detected points highlighted on your chart.

## Examples

### Extract a line chart
```bash
python -m chart2csv.cli.main line_chart.png
```

### Extract with AI (better accuracy)
```bash
export MISTRAL_API_KEY=your_key
python -m chart2csv.cli.main chart.png --use-mistral
```

### Process multiple charts
```bash
python -m chart2csv.cli.main charts_folder/ --batch --output-dir results/
```

### Force chart type
```bash
python -m chart2csv.cli.main chart.png --chart-type scatter
```

## Output Format

CSV output example:
```csv
x,y
0,10
5,25
10,42
15,38
20,55
```

## Next Steps

- [CLI Reference](CLI-Reference) — All command options
- [Troubleshooting](Troubleshooting) — If something doesn't work
- [FAQ](FAQ) — Common questions
