# Contributing to Chart2CSV

Thank you for your interest in contributing to Chart2CSV! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

---

## Code of Conduct

This project follows a standard open-source code of conduct:
- Be respectful and inclusive
- Welcome newcomers
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Tesseract OCR (for CV pipeline)
- A Mistral API key (for LLM features)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Chart2CSV.git
   cd Chart2CSV
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/KikuAI-Lab/Chart2CSV.git
   ```

---

## Development Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e .
pip install -r requirements.txt
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically run linters (black, ruff, mypy) before each commit.

### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env and add your MISTRAL_API_KEY
```

### 5. Verify Installation

```bash
# Run tests
pytest

# Start API server
cd api
python main.py
# Visit http://localhost:8000/docs
```

---

## Making Changes

### Branching Strategy

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Use descriptive branch names:
   - `feature/add-pie-chart-support`
   - `fix/ocr-crash-on-rotated-images`
   - `docs/improve-api-examples`
   - `refactor/extract-common-logic`

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add support for pie charts

fix(ocr): handle rotated images correctly

docs(readme): add installation instructions for Windows

refactor(api): extract common extraction logic to helper function
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=chart2csv --cov-report=html

# Run specific test file
pytest chart2csv/tests/test_mistral.py

# Run tests matching pattern
pytest -k "test_extract"
```

### Writing Tests

- Place tests in `chart2csv/tests/`
- Name test files `test_*.py`
- Use descriptive test names: `test_extract_handles_rotated_images`
- Include unit tests for new functions
- Include integration tests for new features

**Example Test:**

```python
import pytest
from chart2csv.core.pipeline import extract_chart

def test_extract_scatter_chart():
    result = extract_chart("fixtures/scatter_simple.png")

    assert result.chart_type.value == "scatter"
    assert len(result.data) > 0
    assert result.confidence.overall() > 0.5
```

### Test Coverage

- Aim for 70%+ coverage for new code
- Critical paths should have 100% coverage
- Check coverage report: `open htmlcov/index.html`

---

## Code Style

### Automatic Formatting

Pre-commit hooks will automatically format code, but you can run manually:

```bash
# Format with Black
black .

# Sort imports
isort .

# Lint with Ruff
ruff check . --fix

# Type check with mypy
mypy chart2csv
```

### Style Guidelines

1. **Line Length:** 100 characters (configured in pyproject.toml)

2. **Type Hints:** Use type hints for all function signatures
   ```python
   def extract_chart(
       image_path: Union[str, Path],
       chart_type: Optional[ChartType] = None
   ) -> ChartResult:
       ...
   ```

3. **Docstrings:** Use Google-style docstrings
   ```python
   def extract_scatter_points(image: np.ndarray) -> tuple[np.ndarray, float]:
       """
       Extract scatter plot data points from image.

       Args:
           image: Input image array (BGR format)

       Returns:
           Tuple of (points, confidence) where:
           - points: Nx2 array of (x, y) pixel coordinates
           - confidence: Detection confidence (0.0-1.0)

       Raises:
           ValueError: If image is empty or invalid format
       """
       ...
   ```

4. **Naming Conventions:**
   - Functions/variables: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
   - Private members: `_leading_underscore`

5. **Imports:** Group in order:
   - Standard library
   - Third-party packages
   - Local imports

6. **Error Handling:**
   - Raise specific exceptions
   - Include helpful error messages
   - Log errors with context

---

## Submitting Changes

### Before Submitting

1. **Update from upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests:**
   ```bash
   pytest
   ```

3. **Run linters:**
   ```bash
   pre-commit run --all-files
   ```

4. **Update documentation** if needed

### Creating a Pull Request

1. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. **PR Description Should Include:**
   - **Summary:** What does this PR do?
   - **Motivation:** Why is this change needed?
   - **Testing:** How was it tested?
   - **Screenshots:** If UI/output changes
   - **Breaking Changes:** If any

**Example PR Template:**

```markdown
## Summary
Add support for pie chart extraction

## Motivation
Users frequently request pie chart support (#123)

## Changes
- Added `PieChartExtractor` class
- Updated `detect_chart_type()` to recognize pie charts
- Added 15 unit tests for pie chart edge cases

## Testing
- [x] All tests pass
- [x] Added new tests for pie charts
- [x] Tested manually with 20 sample images
- [x] Coverage increased from 65% to 72%

## Breaking Changes
None

## Screenshots
![Pie chart extraction example](docs/pie_chart_example.png)
```

### Review Process

- Maintainers will review your PR
- Address feedback by pushing new commits
- Once approved, PR will be merged

---

## Reporting Bugs

### Before Reporting

1. Check if the bug is already reported in [Issues](https://github.com/KikuAI-Lab/Chart2CSV/issues)
2. Test with the latest `main` branch
3. Gather reproduction steps

### Bug Report Template

```markdown
**Describe the bug**
Clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. Upload image '...'
3. See error

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.5]
- Chart2CSV version: [e.g., 0.1.0]

**Sample image:**
Attach the chart image that triggers the bug

**Error message:**
```
Paste full error traceback here
```

**Additional context**
Any other relevant information.
```

---

## Suggesting Enhancements

We welcome feature suggestions! Please:

1. **Check existing issues** for similar requests
2. **Open a new issue** with the `enhancement` label
3. **Describe the feature:**
   - What problem does it solve?
   - Who would benefit?
   - Proposed implementation (if you have ideas)
   - Examples from other tools

**Example Enhancement Request:**

```markdown
**Feature:** Add support for heatmap extraction

**Problem:** Users with scientific papers often need to extract data from heatmaps

**Proposed Solution:**
- Detect colorbar to map colors → values
- Segment grid cells
- Extract value for each cell
- Return as 2D array

**Alternatives Considered:**
- Manual calibration per cell (too tedious)
- OCR on colorbar (unreliable)

**Priority:** Medium

**Willing to contribute:** Yes
```

---

## Project Structure

Understanding the codebase:

```
chart2csv/
├── core/                  # Core extraction logic
│   ├── pipeline.py        # Main extraction orchestrator
│   ├── llm_extraction.py  # LLM-based extraction
│   ├── detection.py       # Axis/tick detection
│   ├── ocr.py             # OCR for tick labels
│   ├── extraction.py      # Point extraction
│   ├── transform.py       # Pixel → value mapping
│   ├── types.py           # Data structures
│   └── ...
├── cli/                   # Command-line interface
└── tests/                 # Unit tests

api/                       # FastAPI REST API
    └── main.py            # API endpoints

scripts/                   # Development utilities
deploy/                    # Deployment configs
```

---

## Key Concepts

### Extraction Pipeline

```
Image → Preprocess → Detect → OCR → Transform → Extract → Result
```

1. **Preprocess:** Resize, enhance contrast, denoise
2. **Detect:** Find axes and tick marks
3. **OCR:** Read tick labels (Tesseract or Mistral)
4. **Transform:** Build pixel→value mapping
5. **Extract:** Detect and extract data points
6. **Result:** Package with confidence scores

### Confidence Tracking

Every decision includes a confidence score (0.0-1.0):
- Crop detection confidence
- Axis detection confidence
- OCR success rate
- Point extraction confidence

Overall confidence is a weighted average.

### LLM vs CV Mode

- **LLM Mode:** Mistral Pixtral directly extracts data (fast, 90%+ accuracy)
- **CV Mode:** Traditional computer vision pipeline (fallback, works offline)

---

## Getting Help

- **Documentation:** Check the [Wiki](https://github.com/KikuAI-Lab/Chart2CSV/wiki)
- **Issues:** Search [existing issues](https://github.com/KikuAI-Lab/Chart2CSV/issues)
- **Discussions:** Use [GitHub Discussions](https://github.com/KikuAI-Lab/Chart2CSV/discussions) for questions

---

## License

By contributing, you agree that your contributions will be licensed under the AGPL-3.0 License.

---

**Thank you for contributing to Chart2CSV!** 🎉
