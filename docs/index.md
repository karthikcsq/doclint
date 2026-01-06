# DocLint

**ESLint for AI Knowledge Bases**

DocLint detects data quality issues before they cause AI agents to hallucinate or give wrong answers.

## Features

- ✅ Conflict Detection: Identifies contradictory information
- ✅ Staleness Detection: Flags outdated documents
- ✅ Completeness Detection: Validates metadata and content
- ✅ Multi-Format Support: PDF, DOCX, HTML, Markdown, Plain Text
- ✅ Beautiful CLI: Rich terminal output
- ✅ Smart Caching: Fast subsequent scans

## Quick Start

```bash
# Install
pip install doclint

# Scan your knowledge base
doclint scan ./docs/

# Get help
doclint --help
```

## Documentation

- [Getting Started](getting-started.md)
- [Configuration](configuration.md)
- [Detectors](detectors.md)

## Project Status

DocLint is currently in active development (v0.1.0 - Phase 1 Complete).

## License

MIT License - See LICENSE file for details.
