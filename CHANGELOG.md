# Changelog

- 실제 HTML 중간 피드백 `model-feedback` / `coordinate-feedback` 추가 및 기본 실행 경로 변경.
  단일 CSS 속성 변경 후 실제 DOM 전체 박스를 다시 읽어 후보 선택/다음 모델 입력에 사용.
  auto height/줄바꿈/형제 이동을 유지하며 boxes/이름 붙은 frames/전체 raster 비용 모드와 단계별 trace 지원.

- 실제 HTML `web-prepare` / `web-repair` / `web-evaluate` 파이프라인과 일괄 실행 스크립트 추가.
  공유 초기 HTML의 VLM visual self-revision, IR-to-DOM 좌표 변화 반영, 독립 Chromium 평가와
  공식 Design2Code metric 코드 연동(렌더러 통제 변형), 실패·시간·provenance 기록을 지원.

All notable changes are recorded here. This project follows Semantic Versioning
after the first public release.

## [Unreleased]

### Added

- Batched fixed-topology VLM revision baseline, screenshot/frame input modes, multi-round feedback,
  resumable artifacts, retained failures and shared initial-IR evaluation with denoiser baselines.
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
