# TUIDE / FrameDiff — 웹 UI 레이아웃 복원

[![CI](https://github.com/uisanglee/frame-diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/uisanglee/frame-diffusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

목표 UI와 현재 HTML/CSS를 조건으로 수정할 CSS 선언과 값을 예측하고,
브라우저에서 편집·렌더링을 반복하는 연구 코드입니다.
현재 주 실험은 **CSS 우선 적용 WebUI 전처리 → Detector 및 TUIDE 학습 →
저장된 VLM Initial의 레이아웃 복원 → 품질·비용 평가**입니다.
패키지 및 CLI 이름은 `framediff`를 유지합니다.

## 설치

Python 3.11 이상을 사용합니다. 모델별 별도 환경이 필요한 WebSight와
Design2Code-18B 설치는 [전용 모델 안내](docs/screenshot-specialist-models.md)를 따릅니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,visual,web,vlm,visual-metrics]'
python -m playwright install chromium
python -m framediff --help
```

Design2Code 공식 지표의 추가 의존성과 실행 방법은
[평가 안내](docs/paper-evaluation.md)에 있습니다.

## 현재 데이터와 전처리

현재 진입점은 `scripts/prepare_webui_css_color_images.sh`입니다.
기존 `data/new/7000/source/manifest.jsonl`의 페이지 split과 다운로드 캐시를
재사용하고, 새 결과는 `data/new-css-v1`에 저장합니다.

1. 원본 HTML에 외부 CSS를 먼저 적용하고 로컬에 저장합니다.
2. CSS 적용 후 이미지 크기를 확인하여 색상별 `rick` 이미지로 치환합니다.
3. 목표 화면, Detector 라벨, 정규화된 HTML/CSS를 생성합니다.
4. Train의 정상 상태와 Validation·Test의 고정 혼합 손상 데이터를 게시합니다.

```bash
bash scripts/prepare_webui_css_color_images.sh
```

- [전처리 순서·실행·재개·출력 경로](docs/webui-css-first-preparation.md)
- [손상 없음·약한·중간·강한·여러 요소 손상 및 실제 기하 오차 로그](docs/mixed-corruption.md)
- [이전 색상 이미지 데이터에서의 이전 안내](docs/webui-color-images.md)

`7000`은 기존 디렉터리 이름입니다. 실제 페이지 수는 `split-report.json`을
확인해야 합니다. 기존 페이지 split은 유지하며 실패한 페이지를 다른 페이지로
보충하지 않습니다. CSS 적용으로 목표가 바뀐 경우 이전 렌더링·라벨을 새 정답으로
재사용하지 않습니다. 다운로드와 동일한 소스·설정의 검사 캐시는 재사용합니다.

## 학습

현재 학습 경로는 `visual-train-detector`와 `visual-tree-train`입니다.
주 설정은 stylesheet CSS 선언 교체, online mixed corruption,
고정 Validation·Test입니다. S/M/L은 같은 입력·편집 문법을 사용합니다.

```bash
export TRAIN_DATA="$PWD/data/new-css-v1/7000/css"
export TRAIN_RUN="$PWD/runs/webui-css-mixed-v1/7000"

# Detector
CUDA_VISIBLE_DEVICES=2 STAGE=detector \
  bash scripts/run_webui_scale.sh 7000 data/new-css-v1 runs/webui-css-mixed-v1

# Stage1: Detector 없이 학습 가능
CUDA_VISIBLE_DEVICES=0 POLICY_STAGES=1 bash scripts/train_tuide_scale.sh s
```

Stage2는 같은 데이터 버전의 Detector와 예측 target 캐시가 필요합니다.
[학습·캐시·Stage2·S/M/L 비교](docs/tuide-scales-korean.md)를 따릅니다.
[영문 안내](docs/tuide-scales.md)도 제공합니다.

CSS 우선 적용 및 mixed/STOP 학습으로 옮길 때는 새 학습 출력 경로를 사용합니다.
같은 데이터·설정으로 중단된 실행은 기존 `last.pt`와 검사 캐시를 재사용할 수 있습니다.
스크린샷 및 표현 ablation은 `scripts/train_webui_color_ablation.sh`를 사용합니다.

## VLM 벤치마크와 평가

공식 실험 목록은 LLaVA-NeXT-7B, GLM-4.1V-9B, WebSight VLM-8B,
UI2CodeN입니다. WebSight는 image-only 생성이며 Self Revision을 실행하지 않습니다.
다른 모델 및 API 연동 코드는 선택 실행과 기존 결과 재현을 위해 유지합니다.

| 작업 | 실행 파일 / 안내 |
|---|---|
| Design2Code·Hard Initial / Self Revision / UI2CodeN, TUIDE 없이 선행 실행 | `scripts/run_d2c_vlm_baselines.sh`, [실행 안내](docs/d2c-vlm-baselines.md) |
| GPU 0,1 및 2,4의 독립 모델 큐 | `scripts/run_d2c_vlm_parallel.sh` |
| Design2Code·Hard 소규모 BF16 실험 | `scripts/run_d2c_pilot10_bf16_2gpu.sh`, [파일럿 안내](docs/d2c-vlm-pilot.md) |
| WebUI·Design2Code·Hard 통합 실험 | `scripts/run_paper_matrix.sh`, [실험 안내](docs/paper-experiment-suite.md) |
| 세 데이터셋 각각 10개 파일럿 | `scripts/run_paper_pilot10_bf16_2gpu.sh` |
| 저장된 Initial에 S/M/L 적용 | `scripts/run_tuide_scales.sh`, [설정](docs/tuide-scales.md) |
| 품질·실패율·시간 집계 | [평가 안내](docs/paper-evaluation.md) |

WebUI를 포함하는 실행에서는 `WEBUI_TEST`에 새
`data/new-css-v1/7000/css/pages-test.jsonl`을, `TRAIN_RUN` 및 체크포인트에는
동일 데이터 버전의 학습 결과를 명시합니다. 과거 결과 재개 시에는 기존 경로를
그대로 지정하고, 데이터·프로토콜을 변경하면 새 출력 경로를 사용합니다.

모든 복원 방법은 저장된 동일 Initial HTML에서 시작합니다. 출력 상한은
WebUI 16,384, Design2Code·Hard 4,096 토큰이며 현재 파일럿의 Self Revision/UI2CodeN은
2회입니다. 공통 실행 파일에는 `REVISION_ROUNDS=2`를 명시합니다. Initial 생성 실패는 품질 평균에서 제외하고 성공률을 별도
보고합니다. 복원 실패 시 마지막 유효 결과를 유지합니다.

목표는 같은 브라우저 환경에서 정답 HTML을 렌더링한 화면입니다.
Position·CLIP·Box IoU와 생성 성공률을 구분하여 보고하며, 공식 제공 PNG를
그대로 사용한 평가와는 `local-reference-render-v1` 프로토콜로 구분합니다.

## 진단 및 비교 실험

- [Detector·DOM 불일치와 실패 분석](docs/repair-diagnostics.md)
- [정상 복원 경로 및 학습 원인 분석](docs/policy-cause-audit.md)
- [VLM의 첫 악화 편집 분석](docs/vlm-first-edit-audit.md)
- [동일 DOM 손상 강도 비교](docs/structure-severity-audit.md)
- [위치·크기 기반 보호·종료](docs/geometry-preservation.md)
- [Target 출처와 CSS 누락 분석](docs/target-provenance-audit.md)
- [외부 CSS만 재생하는 비교](docs/webui-css-replay.md)

기하 보호는 명시적인 `--preserve-matched-geometry` 옵션입니다. 기존 실험을
재개할 때 자동으로 켜지지 않으며, 비교할 때는 별도 출력 경로를 사용합니다.

## 보존된 독립 실험과 개발 검증

다음은 주 WebUI 학습 경로와 별도인 실험입니다. 공통 자료구조 및 기존
체크포인트와의 호환성 때문에 관련 CLI와 모듈을 유지합니다.

- [합성 CSS 편집 조합](docs/composition-benchmark.md)
- [수치 CSS diffusion](docs/css-diffusion.md)
- [계획 기반 복원](docs/plan-repair.md)
- [LayoutIR 구조](docs/ARCHITECTURE.md)

```bash
python -m pytest -q -m 'not browser'
python -m pytest -q -m browser
make smoke
```

`make smoke`는 합성 LayoutIR 검증이며 TUIDE의 전체 학습·벤치마크가 아닙니다.
`data/`, `runs/`, 다운로드/검사/평가 캐시, 체크포인트는 코드 정리 대상이 아닙니다.
이전 데이터 폴더를 삭제하기 전에는 실제 서버 manifest와 심볼릭 링크의 참조를
확인해야 합니다.

[기여 안내](CONTRIBUTING.md) · [검증 기록](docs/VALIDATION.md) ·
[변경 이력](CHANGELOG.md) · [보안](SECURITY.md)
