# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-13

### Added

- A five-stage Streamlit workflow for data overview, quality review, deterministic
  insights, manual visualisation and optional AI analysis.
- Guarded CSV ingestion with content fingerprints, a 10 MB limit and actionable
  validation errors.
- Safe natural-language chart generation through constrained JSON specifications.
- Line, bar, grouped and stacked bar, scatter, pie, histogram, boxplot, area and
  correlation heatmap rendering.
- Compound categorical and numeric filters, aggregations and chronological sorting.
- Bounded category-level aggregates for factual AI comparisons without sending the
  complete dataset.
- Downloadable AI-assisted analysis reports, responsive mobile behaviour and a
  built-in retail demo dataset.
- Automated compile, lint and test checks for Python 3.11 and 3.12.

### Security

- Model output is validated and never executed as Python, R, JavaScript or SQL.
- Secrets remain outside source control and uploaded data is not persisted.

[1.0.0]: https://github.com/liyang6620/ai-data-analysis-workspace/releases/tag/v1.0.0
