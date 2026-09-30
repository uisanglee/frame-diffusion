# Changelog

- 목표 박스 VLM에 목표 screenshot과 현재 DOM에서 결정적으로 만든 named-frame ID 지도를 함께 입력.
  기존 단일 이미지 prepare 캐시와 섞이지 않도록 prepare protocol 갱신.

- Design2Code 정답 DOM의 controlled CSS corruption을 이용한 VLM/Oracle target box ×
  coordinate/FrameDiff 2×2 진단 파이프라인과 분해 지표를 추가.

- 실제 HTML VLM 출력 검증 재시도, 누락 ID 피드백, 시도별 로그/비용 집계 및 실패 준비 캐시 재처리 추가.
  전체 평균과 모든 방법의 공통 성공 페이지 평균을 별도 보고하고 HTML repair 로그에 오류를 표시.

- WebUI 원본 screenshot/AX/box 파일의 실제 HTML 평가 adapter 추가. `web-prepare --dataset webui`,
  뷰포트 선택, HTML 없는 입력, 독립 recorded-box 평가, 누락 metadata 기록을 지원.

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
