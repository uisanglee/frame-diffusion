# FrameDiff — 웹 컴포넌트 위치·크기 보정 연구 코드

[![CI](https://github.com/uisanglee/frame-diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/uisanglee/frame-diffusion/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11–3.14](https://img.shields.io/badge/python-3.11--3.14-blue.svg)](pyproject.toml)

실행 가능한 레이아웃 트리에 노이즈를 넣고, 목표 frame을 조건으로 속성 편집을 반복하는 **diffusion-inspired tree mutation denoiser**입니다. 데이터 생성 → 학습 → 체크포인트 → 수정 → 비교 실험 → Chromium 검증 → 결과 집계까지 포함합니다.

> Status: research prototype (`v0.1.0`). API, LayoutIR schema, checkpoint compatibility may change before `v1.0`.

## 저장소 안내

- [GitHub 운영·최초 게시 가이드](docs/GITHUB.md)
- [기여 및 연구 재현 규칙](CONTRIBUTING.md)
- [로컬 검증 기록](docs/VALIDATION.md)
- [변경 이력](CHANGELOG.md)
- [보안·데이터 취급](SECURITY.md)
- [실제 HTML Self-Revision vs FrameDiff 전체 실험](docs/REAL_WEB_EXPERIMENT.md)

소스와 설정만 Git으로 관리합니다. `data/`, `runs/`, checkpoint, VLM weight, Playwright cache는 의도적으로 제외됩니다. MIT 라이선스이며 실제 저자·기관 정보는 공개 전에 `CITATION.cff`에 보완하세요.

중요: 원 논문 구현을 그대로 재현한 프로젝트나 임의의 React/CSS를 자동 수정하는 완성품은 아닙니다. 첫 실험의 범위는 **동일 컴포넌트 집합·부모 관계에서 위치, 크기, 배치 속성, 형제 순서 수정**입니다. DDPM의 Gaussian noise/timestep 모델도 아닙니다. 연구 가설을 검증할 수 있는 실행 가능한 출발점입니다.

실제 HTML 실험의 기본값은 이제 `model-feedback`입니다. 한 CSS 속성 수정마다 실제 Chromium layout을
다시 읽어 줄바꿈·auto height·형제 이동을 다음 입력에 반영합니다. `frames`는 실제 박스를 이름 붙은
프레임 그림으로 저장하지만, 현재 모델은 이미지 픽셀이 아니라 측정 박스/트리 특징을 입력받습니다.
기존 `model`은 proxy-only ablation으로 유지합니다. [실행·제약·비용 비교](docs/REAL_WEB_EXPERIMENT.md)를 참고하세요.
같은 정답 DOM을 교란한 뒤 VLM/Oracle 목표 박스와 coordinate/FrameDiff 정책을 교차하는
2×2 진단은 `scripts/run_oracle_ablation.sh`로 실행합니다.

WebUI 원본 test 아카이브도 `WEB_DATASET=webui`로 같은 실제 HTML 파이프라인에 연결할 수 있습니다.
저장 screenshot에서 초기 HTML을 생성하며, 평가는 원본 screenshot/AX 박스를 사용합니다.
자세한 명령과 지표 정의는 [WebUI 실행 가이드](docs/REAL_WEB_EXPERIMENT.md#webui-원본-다운로드로-실행)에 있습니다.

## 1. 바로 실행

Python 3.11–3.14 사용. 이 작업 폴더에는 Python 3.12의 `.venv`가 준비되어 있습니다.

### Conda

저장소 root에서 다음을 실행합니다. `environment.yml`은 정확한 lock file이 아니라 Python과
프로젝트 의존성을 설치하는 재현 가능한 진입점입니다.

```bash
conda env create -f environment.yml
conda activate framediff
python -m playwright install chromium
python -m pytest -q
```

의존성을 변경한 뒤 기존 환경을 맞추려면:

```bash
conda env update -f environment.yml --prune
```

### venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[browser,test]'
python -m playwright install chromium
python -m framediff generate --out data/smoke --count 150 --seed 42
python -m framediff suite --config configs/smoke.json
python -m pytest -q -m 'not browser'
python -m pytest -q -m browser
```

이 컴퓨터에 이미 설치된 Chromium을 쓰려면:

```bash
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/ms-playwright"
```

`suite`는 학습, 비교 평가, seed별 집계를 순서대로 실행합니다. `runs/smoke-suite/aggregate.json`이 최종 결과입니다. 50-step smoke 설정은 연결 검증용이지 성능 평가용이 아닙니다.

## 2. RTX 4090 학습·실험

Linux에서 GPU/드라이버와 호환되는 CUDA PyTorch를 먼저 설치하고 `torch.cuda.is_available()`을 확인하세요. 아래 설정의 메모리 사용량은 이 Mac에서 검증하지 않았습니다.

Conda를 사용한다면:

```bash
conda env create -f environment-4090.yml
conda activate framediff-4090
python -m playwright install --with-deps chromium
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

마지막 값이 `True`가 아니면 학습을 시작하지 말고, 해당 서버의 NVIDIA driver에 맞는 PyTorch
wheel을 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)로 다시 설치하세요.
Conda 환경 자체가 CUDA driver를 설치하지는 않습니다. VLM까지 같은 환경에서 실행하려면
`python -m pip install -e '.[vlm]'`을 추가하되 24GB VRAM에서 학습 모델과 동시에 적재하지 않습니다.

```bash
python -m pip install -e '.[browser,test]'
python -m playwright install --with-deps chromium
python -m framediff generate --out data/synthetic --count 10000 --seed 42
python -m framediff suite --config configs/4090.json
```

4090 설정: hidden=512, 8 layers, 8 heads, batch=8, accumulation=4, BF16, 10,000 steps, 3 seeds. VLM은 동시에 올리지 않습니다. OOM이면 `configs/4090.json`의 batch를 4→2로 낮추고 accumulation을 올리세요. 학습 시간·VRAM 수치를 측정 전 보장하지 않습니다.

단일 학습과 재개:

```bash
python -m framediff train \
  --train data/synthetic/train.jsonl --val data/synthetic/val.jsonl \
  --out runs/main --device cuda --bf16 --hidden 512 --layers 8 \
  --batch-size 8 --accumulation 4 --steps 10000

# 위 명령에 --resume runs/main/last.pt 를 추가하고 동일 구조 설정 유지
```

`best.pt`, `last.pt`, `train.jsonl`, `manifest.json` 저장. optimizer와 RNG를 복구합니다. 데이터 변경 후 resume하지 마세요. validation의 편집 loss로 checkpoint를 고르며 test 점수로 선택하지 않습니다.

## 3. 실행 구조

1. AR VLM 또는 기존 시스템이 초기 LayoutIR 생성.
2. 목표 frame을 `(id, role, name, parent, x, y, width, height)`에 대응시킴.
3. 트리 속성/역할/이름 hash/부모 관계와 현재·목표 box를 denoiser에 입력.
4. 모델이 합법적인 `(node, property, new_value)` 편집 후보 제안.
5. 경량 CPU layout executor로 후보 평가, beam search로 선택.
6. 마지막에만 HTML로 변환하고 Chromium으로 실제 좌표 검증.

모델 루프에는 이미지 인코더가 **없습니다**. 이름을 그린 frame 이미지를 매번 인코딩하지 않고 동일 정보를 구조화된 tensor로 입력합니다. 역할 embedding과 이름 hash는 학습되지만, 이름 hash가 pretrained 언어 의미를 제공하는 것은 아닙니다. 스크린샷만 주어지면 별도 frozen VLM이 frame을 한 번 추출합니다.

학습은 clean tree에 1–5회의 유효 속성 mutation을 적용하고, clean 값으로 되돌리는 편집들의 확률 합을 최대화합니다. 15%는 STOP 예제입니다. 이는 multi-step corruption에서 복원 편집을 학습하는 방식이며, 원 논문의 subtree replacement 분포/역과정 likelihood 재현은 아닙니다. 탐색은 관측 frame 오차를 사용하고 STOP/value head를 종료·선택 기준으로 쓰지 않습니다.

## 4. 데이터 선택

### A. 합성 실행 트리 — 바로 학습 가능

`generate`는 header/nav/sidebar/card/grid/footer 등의 제한된 DSL 트리를 만듭니다. 각 clean tree, 손상된 current tree, 3개 viewport 목표 box를 저장합니다. viewport는 1024×768, 768×1024, 390×844입니다. 같은 속성 코드를 여러 폭에서 실행할 뿐, 미디어 쿼리나 모바일 재배치는 아직 없습니다. 작은 화면 overflow도 데이터에 남고 metric으로 보고됩니다.

현재 split은 생성 seed 단위입니다. 같은 템플릿 계열과 ID 이름이 train/test에 재사용되므로, 이 점수는 **in-distribution 복원**만 의미합니다. 논문 수준의 일반화 주장을 위해 템플릿·사이트·출처가 분리된 실데이터 테스트가 추가로 필요합니다.

### B. WebUI — 관측 frame 기반 geometry 사전학습

[공식 WebUI 저장소](https://github.com/js0nwu/webui)의 원본 아카이브를 사용자가 라이선스·용량을 확인하고 내려받아 압축 해제합니다. 자동으로 대규모 데이터를 다운로드하지 않습니다. `*axtree.json.gz`, 같은 접두사의 `bb.json.gz`, `url.txt`를 읽습니다. `viewport.json.gz`는 화면 크기가 아니라 노드별 visibility 정보이므로 크기로 오해하지 않습니다.

```bash
python -m framediff import-webui --root /path/to/webui-extracted \
  --out data/webui-raw --max-nodes 127 --limit 1000
python -m framediff fit-frames --input data/webui-raw/observations.jsonl \
  --out data/webui-fit --max-error 0.15
python -m framediff train --train data/webui-fit/train.jsonl \
  --val data/webui-fit/val.jsonl --out runs/webui --device cuda --bf16
```

AX tree와 좌표는 원본 CSS 정답이 아닙니다. `fit-frames`는 absolute-layout surrogate를 만들고 양자화/재부모화 오차를 `fit-report.json`에 기록합니다. 기본적으로 평균 IoU가 0.85 미만인 샘플을 제외합니다. 입력·평가 target은 실제 관측 box이고, 편집 label은 이를 가장 가깝게 재현한 surrogate IR입니다. 이 데이터는 실제 box geometry 사전학습·held-out 평가용이지 원본 웹 CSS 복원 benchmark가 아닙니다.

도메인 hostname 기준 split, 학습/검증 및 평가 시 group 중복 검사를 제공합니다. 공식 split을 보존하려면 import에 `--split train|val|test`를 지정하고 각각 따로 처리하세요. 공식 split 간에도 같은 hostname이 있으면 의도적으로 실패하므로 정리해야 합니다. 노드 초과는 조용히 자르지 않고 거절합니다.

### C. 자체 AR 오류 데이터

실제 목적에 가장 가까운 것은 `reference → AR initial IR → clean executable IR` 쌍입니다. corruption-only와 AR-error fine-tuning을 분리해 비교하세요.

```bash
python -m framediff make-pair --current initial.json --target target.json \
  --clean clean.json --id page-001 --group example.org \
  --target-kind oracle_frames --out pair.jsonl
```

여러 JSONL을 도메인별로 묶어 train/val/test를 구성합니다. `--fixed-pairs` 학습은 저장된 AR 오류를 사용하고, 기본 학습은 clean에 온라인 corruption을 적용합니다. 부모 관계나 노드 집합이 다르면 명시적으로 거절합니다. 이름만으로 객체 대응을 자동으로 추측하지 않습니다.

### D. Design2Code / Design2Code-Hard — 실제 페이지 최종 평가

[공식 Design2Code 저장소](https://github.com/SALT-NLP/Design2Code)의 데이터는 각각 같은 이름의
`xx.png`와 `xx.html` 쌍입니다. 이 프로젝트가 데이터를 자동 재배포하지 않으므로 공식 배포처에서
Design2Code(484개)와 Hard(80개)를 내려받아 압축을 풉니다.

먼저 JavaScript와 네트워크를 차단한 Chromium에서 reference HTML의 visible DOM box를 추출합니다.
동시에 reference box를 근사하는 oracle surrogate test도 만듭니다.

```bash
python -m framediff prepare-design2code \
  --root /datasets/Design2Code/testset_final \
  --out data/design2code --dataset design2code --max-nodes 127

python -m framediff prepare-design2code \
  --root /datasets/Design2Code-Hard \
  --out data/design2code-hard --dataset design2code-hard --max-nodes 127
```

`prepare-report.json`에서 거절 사유, visible node 수, surrogate fit IoU를 확인하세요. `manifest.jsonl`은
실제 DOM box 정답이고 `oracle-test.jsonl`은 동일 ID의 actual-box oracle 평가입니다.

```bash
python -m framediff evaluate --data data/design2code/oracle-test.jsonl \
  --checkpoint runs/main/best.pt --out runs/design2code-oracle --device cuda \
  --methods none,coordinate,random,model,copy-boxes --browser-verify
```

실제 screenshot-to-repair 평가는 Qwen을 한 번 적재한 뒤 각 페이지에서 initial LayoutIR와 그 IR ID에
대응한 predicted target frame을 생성합니다. `--resume`은 이미 저장된 두 VLM 산출물을 재사용합니다.

```bash
python -m framediff run-design2code \
  --manifest data/design2code/manifest.jsonl \
  --out data/design2code-qwen --backend qwen --four-bit --resume

python -m framediff evaluate --data data/design2code-qwen/test.jsonl \
  --checkpoint runs/main/best.pt --out runs/design2code-e2e --device cuda \
  --methods none,copy-boxes,coordinate,model --browser-verify --screenshots
```

`none`은 AR initial IR, `copy-boxes`는 VLM이 예측한 frame 자체, `model`은 FrameDiff 수정 결과입니다.
생성 IR의 ID와 reference DOM ID가 다르므로 평가는 IoU 최대 Hungarian geometry matching을 사용합니다.
이 값은 이 연구의 component geometry 지표이며 Design2Code 공식 CLIP/text/color/block-match 점수를
사칭하지 않습니다. `upstream_seconds`는 두 VLM 호출, `end_to_end_seconds`는 VLM과 수정을 합친 시간입니다.

WebUI와 synthetic을 함께 학습하려면 split별로 명시적으로 합칩니다.

```bash
python -m framediff combine-data --split train \
  --input data/synthetic/train.jsonl --input data/webui-fit/train.jsonl \
  --out data/combined/train.jsonl
python -m framediff combine-data --split val \
  --input data/synthetic/val.jsonl --input data/webui-fit/val.jsonl \
  --out data/combined/val.jsonl
```

WebSight는 이 프로토콜의 필수 평가 데이터가 아니며 포함하지 않습니다.

## 5. VLM 연결 (선택)

기본 adapter는 [Qwen3-VL-8B-Instruct 공식 API](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)에 맞췄습니다. RTX 4090에서는 4-bit 옵션으로 시작하세요. 모델 다운로드·CUDA VLM 추론은 이 Mac에서 실행하지 않았습니다. 정확한 재현에는 `--revision`으로 model commit을 고정하세요.

```bash
python -m pip install -e '.[vlm]'
python -m framediff vlm --backend qwen --four-bit \
  --task generate-ir --image reference.png --out initial.json \
  --max-new-tokens 8192
python -m framediff vlm --backend qwen --four-bit \
  --task extract-frames --image reference.png --current initial.json \
  --out estimated-target.json
python -m framediff vlm --backend qwen --four-bit \
  --task revise-ir --image reference.png --current initial.json \
  --out ar-revised.json --max-new-tokens 8192
```

`generate-html` / `revise-html`도 지원하지만, 임의 HTML의 자동 CSS 역변환은 제공하지 않습니다. `extract-html`은 self-contained static HTML의 실제 frame을 추출할 뿐입니다. 네트워크와 페이지 JavaScript를 차단하므로 외부 CSS/폰트/React hydration이 필요한 페이지는 동일하게 재현되지 않습니다.

로컬 또는 원격의 호환 API 서버:

```bash
python -m framediff vlm --backend openai-compatible \
  --endpoint http://localhost:8000/v1/chat/completions --model YOUR_MODEL \
  --task revise-ir --current initial.json --image reference.png --out revised.json
```

키는 선택적으로 `VLM_API_KEY` 환경변수를 읽습니다. 원격 endpoint는 이미지·코드를 전송하므로 민감 자료에 주의하세요. 기본값은 localhost이며 자동 외부 전송은 없습니다. VLM 원문, prompt, 시간, 가능한 경우 token usage를 output 옆에 저장합니다. 형식 오류는 자동으로 정답을 고치거나 버리지 않고 실패합니다.

VLM을 **유일한 평가 심판으로 사용하지 않습니다**. 일차 평가는 ID 대응된 좌표와 실제 브라우저 결과입니다. 예측 frame 실험은 독립적인 실제 정답을 반드시 제공해야 합니다:

```bash
python -m framediff make-pair --current initial.json --target estimated-target.json \
  --ground-truth actual-target.json --target-kind predicted_frames \
  --id page-001 --group example.org --out predicted-pair.jsonl
```

`evaluation_observations`는 점수 계산에만 쓰고 모델·탐색에는 전달하지 않습니다. 입력으로 제공한 추정 좌표를 그대로 복사하는 `copy-boxes`도 이때는 독립 정답과 평가합니다.

## 6. 수정과 렌더링

VLM 직접 수정 baseline은 먼저 별도 프로세스에서 생성한 뒤 동일 평가기에 넣습니다.
Qwen과 FrameDiff를 동시에 GPU에 적재할 필요가 없습니다.

```bash
python -m framediff vlm-baseline \
  --data data/design2code-qwen/test.jsonl --out data/design2code-vlm-baseline \
  --backend qwen --four-bit --rounds 1 --resume

python -m framediff evaluate \
  --data data/design2code-vlm-baseline/test.jsonl \
  --checkpoint runs/main/best.pt --out runs/design2code-baselines --device cuda \
  --methods none,vlm-revise,coordinate,model \
  --steps 10 --beam 2 --topk 8 --budget 160 --browser-verify --screenshots
```

Hard는 경로의 `design2code`를 `design2code-hard`로 변경합니다.
WebUI는 `--data data/webui-fit/test.jsonl --out data/webui-vlm-baseline`을 사용합니다.
처음에는 생성 명령에 `--limit 5`를 추가해 확인하고 전체 실행 시 제거하세요.

- `none`: 동일 AR 초기 IR (WebUI는 corrupted surrogate).
- `vlm-revise`: VLM 직접 수정. 기본 1회; `--rounds 3`은 매번 현재 결과를 렌더링해 총 3회 수정.
- `model`: 동일 초기 IR에서 FrameDiff 수정. VLM baseline의 결과로 시작하지 않습니다.
- screenshot이 있으면 목표 screenshot + 현재 frame screenshot + 현재 IR를 입력합니다.
  `--input-mode frames` 또는 screenshot이 없는 WebUI에서는 입력 target boxes + 현재 frame screenshot + IR를 제공합니다.
  입력 정보가 다른 프로토콜이므로 screenshot 조건과 oracle-box 조건은 별도로 보고하세요.
- reference HTML, clean IR, 독립 evaluation box는 VLM에 전달하지 않습니다.
- FrameDiff와 동일하게 node ID/부모/순서는 고정합니다. 자유로운 HTML 재생성 baseline은 아닙니다.
- `repair_seconds`에 VLM 수정 및 피드백 rendering 시간이 포함됩니다. 모델 로딩은 제외됩니다.
- `vlm_failure_rate`와 `status.json`을 함께 보고하세요. 실패 페이지를 제외하지 않고 마지막 유효 IR를 평가합니다.
- `--resume`은 입력/설정 hash가 같은 산출물만 재사용합니다. 실패 산출물도 재사용하므로 재시도는 새 출력 폴더를 쓰세요.
- 새 Design2Code 생성 레코드는 initial generation/frame extraction 시간을 분리합니다.
  이전 레코드의 `end_to_end_seconds`는 공통 upstream 비용을 포함하므로 VLM-only 시간으로 해석하지 마세요.
  이전 데이터에서는 `repair_seconds`로 수정 비용을 비교하거나 `run-design2code --resume`으로 레코드를 재구성하세요.

출력 JSON/CSV의 `browser_box_iou`가 실제 Chromium 기준 정확도입니다.
기본 `report.md`의 IoU 열은 proxy 좌표 기준이므로 browser 검증 값과 구분하세요.

```bash
python -m framediff export-example --data data/smoke/test.jsonl --out runs/example
python -m framediff repair --ir runs/example/current.json \
  --target runs/example/target.json --checkpoint runs/main/best.pt \
  --out runs/repaired --device cuda --steps 10 --budget 160
python -m framediff render --ir runs/repaired/repaired.json \
  --out runs/repaired/render --width 1024 --height 768 --screenshot
```

`repaired.json`, viewport별 HTML, 수정 trace가 생성됩니다. 최종 frame HTML은 디버그용 이름/테두리를 표시합니다. 실제 서비스의 디자인·텍스트·이미지까지 재생성하지 않습니다.

## 7. 비교 실험

```bash
python -m framediff evaluate --data data/synthetic/test.jsonl \
  --checkpoint runs/main/best.pt --out runs/main/eval --device cuda \
  --methods none,coordinate,random,model,copy-boxes \
  --steps 10 --beam 2 --topk 8 --budget 160 --browser-verify

# 매 후보 Chromium 실행 비용 vs 경량 executor (별도 소규모 측정)
python -m framediff evaluate --data data/synthetic/test.jsonl \
  --checkpoint runs/main/best.pt --out runs/main/latency --device cuda \
  --methods model,model-browser --limit 20 --budget 160 --browser-verify
```

| 방법 | 의미 |
|---|---|
| `none` | 입력 코드 그대로 실행 |
| `coordinate` | target-directed 후보 + 유한차분 coordinate/beam search |
| `random` | 같은 비신경 후보 집합에서 무작위 제안 + 오차 기반 선택 |
| `model` | 학습된 편집 제안 + 경량 executor 기반 선택 |
| `model-browser` | 같은 모델, 후보마다 실제 Chromium 좌표로 피드백 |
| `copy-boxes` | 입력 목표 좌표 복사; 실행 코드가 없는 진단용 기준 |
| `one-shot` | `--objective boxes`로 별도 학습한 회귀 head; 단일 viewport만 지원 |

AR baseline은 `vlm --task revise-ir` 출력으로 pair를 만들고 `none`으로 평가합니다. AR 호출 시간은 VLM `.meta.json`의 wall/generation 시간을 합쳐 보고해야 하며 이 평가 러너가 자동으로 더하지는 않습니다. 초기 generation, frame extraction 비용도 end-to-end 비교에 별도로 포함하세요.

`configs/4090.json`은 tree bias 제거 및 fixed-pair 학습 ablation을 포함합니다. no-tree-bias는 관계 attention bias만 제거하며 depth/parent-derived geometry까지 모두 제거하는 것은 아닙니다. one-shot은 단일 viewport 데이터로 따로 학습·평가해야 합니다.

평가 산출물:

- `results.jsonl`: page별 IoU, center/size error, relation, success, overflow, 실행 횟수, 지연.
- `summary.csv/json`, `report.md`: 평균, p95 latency, group-bootstrap IoU CI.
- `predictions/`: 수정 IR, frame, 편집 trace. `--screenshots`는 최종 검증 스크린샷.
- suite의 `aggregate.json`: seed별 평균·표준편차. 반복 seed를 독립 페이지로 취급하지 않음.

후보 budget는 모델 forward 수가 아니라 제안된 후보 수입니다. 실제 execution은 캐시, viewport 수, early stopping에 따라 달라집니다. `proxy_executions`는 탐색 scorer의 실행 횟수이며 모든 Python 보조 계산 호출의 프로파일러는 아닙니다. `repair_seconds`는 checkpoint 로딩을 제외한 수정 단계, `total_seconds`는 최종 좌표 검증·metric 계산까지 포함하되 결과 파일 쓰기는 제외합니다. browser 실행 횟수와 이미지 screenshot 횟수는 다른 비용입니다. 이 코드의 model-browser는 geometry 재실행 비교이지 매번 raster+VLM을 돌리는 비교가 아닙니다.

LayoutDM, LayoutFormer++, 원 논문의 공개 결과를 이 내부 baseline 이름으로 바꾸어 비교하지 않습니다. 탑티어 논문과의 정식 비교에는 각 공식 구현, 같은 데이터·출력 제약·관측 정보·계산 예산을 맞춘 별도 재현이 필요합니다.

## 8. IR 명세와 한계

경량 executor의 속도 이득과 정확도 손실은 `compare-renderers`로 paired 평가합니다.
같은 페이지·checkpoint·candidate budget에서 proxy, Chromium DOM layout, 선택적으로 PNG 렌더링을
비교하며 **모든 최종 IoU는 Chromium 결과로 계산**합니다. Predicted-frame 실험의 독립 reference 정답은
탐색에 전달되지 않습니다. Oracle 실험은 명시적으로 실제 target box를 탐색 입력으로 사용합니다.

```bash
python -m framediff compare-renderers \
  --data data/synthetic/test.jsonl --checkpoint runs/main/best.pt \
  --out runs/render-cost/synthetic --device cuda \
  --limit 20 --repeats 3 --warmup 1 --steps 10 --beam 2 --topk 8 --budget 160 --raster
```

동일 명령의 `--data`와 `--out`을 변경하여 각 데이터셋을 별도로 평가합니다.

| Dataset | `--data` | `--out` |
|---|---|---|
| WebUI held-out | `data/webui-fit/test.jsonl` | `runs/render-cost/webui` |
| Design2Code screenshot 입력 | `data/design2code-qwen/test.jsonl` | `runs/render-cost/design2code` |
| Design2Code-Hard screenshot 입력 | `data/design2code-hard-qwen/test.jsonl` | `runs/render-cost/design2code-hard` |
| Design2Code oracle 진단 | `data/design2code/oracle-test.jsonl` | `runs/render-cost/design2code-oracle` |

`--limit 0`은 전체 데이터입니다. 20개는 앞부분이 아니라 seed로 무작위 선택하며 sample ID를 저장합니다.
Warm-up 후 각 페이지의 backend 실행 순서를 무작위화하고 각 반복에 새 탐색 캐시를 사용합니다.
`report.md`, `summary.json`, `paired.jsonl`, `results.jsonl` 및 모든 최종 IR/trace를 저장합니다.

- `repair_speedup`: Chromium 탐색 시간 / proxy 탐색 시간.
- `pipeline_speedup`: 저장된 VLM 호출 시간과 최종 Chromium 검증까지 포함한 비율.
- `iou_loss_pp`: 100 × (Chromium 탐색 최종 IoU − proxy 탐색 최종 IoU). 양수면 proxy 손실.
- `iou_loss_pp_ci95`: 반복을 페이지별 평균한 뒤 도메인/페이지 cluster bootstrap 95% 구간.
- `search_browser_calls`, `verification_browser_calls`: 탐색과 최종 검증 호출을 분리.
- `parity_max_error_px`: 같은 최종 IR에 대한 proxy/Chromium 좌표 차이.
- `search_screenshots`: raster 옵션에서 실제 생성한 PNG 수. PNG 인코딩 포함, 디스크 저장/VLM 호출 제외.

두 방법 모두 같은 LayoutIR가 컴파일된 frame HTML을 실행합니다. 원본 웹페이지의 전체 CSS·텍스트·asset
렌더링과의 비용 비교는 아닙니다. Design2Code 입력/reference의 재현 품질과 DOM 선택에 따른 오차도 별도입니다.
`--budget`은 상한이므로 early stop과 후보 차이에 따라 실제 후보/호출 수는 달라질 수 있습니다.
모델/브라우저 시작 비용은 제외하며 VLM 실행 시간이 저장되지 않은 데이터의 pipeline 시간은 수정+검증만입니다.

모든 노드는 `id,parent,role,name,props`, 트리는 `version:1,nodes:[...]`입니다. root는 첫 노드이고 viewport 크기로 고정됩니다.

| 속성 | 인코딩 |
|---|---|
| width | 1–256, 부모 content 폭의 value/256 (grid는 cell 폭) |
| height | 1–256, value×4 px |
| dx,dy | 0–256, 일반 flow에서는 (value−32)×4 px 이동 |
| flow | 0=row, 1=column, 2=grid, 3=absolute |
| gap,padding | 0–16, value×4 px |
| columns | 1–6 |
| align | 0=start,1=center,2=end |
| order | 0–31, 형제 순서 |

부모가 absolute면 자식의 dx/dy는 부모 inner 크기의 value/256 위치입니다. absolute fitting의 선택은 geometry 실험 편의를 위한 것으로 반응형 CSS 설계의 정답이 아닙니다.

미지원: 노드 추가/삭제/부모 변경, 임의 CSS selector cascade, auto/intrinsic text size, 글꼴·줄바꿈, assets, media queries, CSS AST source patching, 자유로운 페이지 의미 추론. 이를 감추지 않고 제한된 compiler/executor parity를 먼저 검증하는 설계입니다.

## 9. 코드 위치

- `framediff/ir.py`: IR 검증, 경량 executor, HTML compiler.
- `data.py`, `adapters.py`, `benchmarks.py`: corruption, split, WebUI import/fit, Design2Code batch pipeline.
- `model.py`, `train.py`: Transformer 편집 모델, 학습/resume.
- `search.py`, `evaluate.py`, `suite.py`: 탐색, 비교, seed 집계.
- `browser.py`, `vlm.py`: Chromium 검증, frozen VLM adapter.
- `tests/`: mutation·loss·누수 방지·adapter·proxy/browser parity.
- `environment.yml`, `environment-4090.yml`: CPU/개발 및 4090 학습용 Conda bootstrap 환경.

배경 논문: [Diffusion On Syntax Trees For Program Synthesis, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html). 이 구현의 차이와 실측 검증은 `docs/VALIDATION.md`에 기록합니다.
