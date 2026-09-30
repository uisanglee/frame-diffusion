# Changelog

All notable changes are recorded here. This project follows Semantic Versioning
after the first public release.

## [Unreleased]

### Added

- `compare-renderers`: paired proxy/Chromium/PNG latency and final-browser accuracy experiment,
  warmup, randomized order, repeated measurements and paired cluster-bootstrap confidence intervals.
- GitHub Actions CI, contribution templates, dependency updates, and repository documentation.
- Conda bootstrap environments for local development and RTX 4090 training.
- Design2Code/Hard HTML-box preparation and batched screenshot-to-LayoutIR/frame inference.
- Identity-free Hungarian geometry evaluation against independent reference DOM boxes.
- Actual-box WebUI evaluation, combined-dataset construction, and end-to-end latency accounting.

## [0.1.0] - 2026-09-30

### Added

- Executable LayoutIR and proxy/browser parity validation.
- Synthetic and WebUI-derived data pipelines.
- Tree mutation denoiser training, checkpoint resume, and beam search repair.
- Coordinate, random, copy-box, one-shot, and browser-loop evaluation baselines.
- Optional Qwen3-VL and OpenAI-compatible VLM adapters.
- Multi-seed experiment runner and local smoke validation.
