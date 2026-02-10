# Documentation Build

This directory contains the documentation setup for the TRL project using Sphinx, Doxygen, and Breathe.

## Prerequisites

- Doxygen (for generating API documentation from C++ code)
- Python 3 with virtual environment
- Sphinx, Breathe, and sphinx-rtd-theme (installed in venv)

## Setup

A Python virtual environment is already configured in `docs/venv/` with the necessary packages:
- sphinx
- breathe
- sphinx-rtd-theme

To activate the virtual environment:

```bash
source docs/venv/bin/activate
```

## Building Documentation

From the `docs/` directory:

### Generate all documentation (Doxygen + Sphinx):
```bash
make all
```

### Generate only Doxygen XML:
```bash
make doxygen
```

### Generate only Sphinx HTML:
```bash
make html
```

### Clean generated files:
```bash
make clean
```

## Viewing Documentation

After building, open `docs/_build/html/index.html` in your web browser.

## Project Structure

- `Doxyfile` - Doxygen configuration (in project root)
- `docs/conf.py` - Sphinx configuration
- `docs/index.rst` - Main documentation page
- `docs/api/index.rst` - API reference page
- `docs/doxygen/xml/` - Generated Doxygen XML (used by Breathe)
- `docs/_build/html/` - Generated HTML documentation
