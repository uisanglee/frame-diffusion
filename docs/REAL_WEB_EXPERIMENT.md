# 실제 HTML Self-Revision vs FrameDiff

이 파이프라인은 프레임 그림이 아닌 **실제 HTML/CSS 페이지**를 생성·수정·렌더링합니다.
기존 `run-design2code` / `vlm-baseline`의 LayoutIR 전용 실험과 출력 형식이 다릅니다.

## WebUI 원본 다운로드로 실행

`webui-fit/test.jsonl` 대신 **압축 해제한 WebUI test 원본 폴더**를 사용합니다.
`*-screenshot.webp`(또는 png), 대응하는 `*-axtree.json.gz`, `*-bb.json.gz`, `*-url.txt`를
자동으로 연결합니다. 저장 HTML은 선택 사항입니다. 초기 HTML은 screenshot에서 VLM이 새로 생성하며,
그 동일한 초기 HTML을 Self-Revision / FrameDiff가 각각 수정합니다.

기존 설치 환경(`.[web,vlm]`, Chromium)에서 먼저 5개:

```bash
# 실제 압축 해제된 test 폴더로 설정. train/val까지 포함한 상위 폴더를 지정하지 마세요.
export WEBUI_TEST_ROOT=/datasets/webui/test

WEB_DATASET=webui WEBUI_VIEW=default_1280-720 \
PAGE_LIMIT=5 REVISION_ROUNDS=1 \
bash scripts/run_real_web.sh \
  "$WEBUI_TEST_ROOT" runs/webui-feedback-smoke runs/main/best.pt
```

전체 실행은 별도 출력 폴더에서:

```bash
WEB_DATASET=webui WEBUI_VIEW=default_1280-720 \
REVISION_ROUNDS=3 \
bash scripts/run_real_web.sh \
  "$WEBUI_TEST_ROOT" runs/webui-feedback-full runs/main/best.pt
```

`WEBUI_VIEW`는 정확한 접두사입니다. 다른 뷰포트는 `default_1920-1080`처럼 지정하거나 `all`을 사용합니다.
기본은 페이지당 `default_1280-720` 하나로 평가합니다. `all`은 같은 사이트의 여러 화면을 각각 별도 사례로
처리하는 것이며, 하나의 HTML을 모든 viewport에 공동 최적화하는 실험이 아닙니다.
`*-screenshot-full.webp`는 사용하지 않습니다. WebUI의 viewport JSON은 visibility metadata이므로
화면 크기로 오해하지 않고 screenshot 크기/파일명과 비교합니다. 좌표 배율을 확인할 수 없는 화면은
박스 평가 오류를 기록하되 screenshot 평가와 페이지 자체는 유지합니다.

단계별 prepare 명령은 다음과 같습니다. 이후 repair/evaluate 명령은 아래 Design2Code 예의 경로만 바꾸면 됩니다.

```bash
python -m framediff web-prepare --dataset webui \
  --root "$WEBUI_TEST_ROOT" --webui-view default_1280-720 \
  --out runs/webui-feedback-smoke/prepare --backend qwen --four-bit \
  --rounds 1 --limit 5 --resume
```

WebUI 결과의 `evaluation/report.md`, `summary.json`에서 확인할 항목:

- `pixel_mae`: 생성 HTML의 최종 screenshot과 **저장된 원본 screenshot**의 차이.
- `webui_box_iou`, `webui_center_error`, `webui_size_error`: 기록된 AX 박스와 생성 DOM 박스를 viewport로
  잘라 Hungarian matching으로 비교한 **보조 기하 지표**. AX/DOM의 요소 구분이 달라 완전한 정답 대응은 아닙니다.
- `pipeline_seconds`, `pipeline_browser_executions`, `vlm_calls`: 초기 생성·목표 박스 추출·중간 실제 렌더링을 포함한 비용.
- `failure_rate`, `reference_failures`, `webui_box_iou_n`: 수정 실패와 참조 박스 누락을 구분합니다.
  AX/박스 파일이 없거나 깨져도 screenshot이 있으면 해당 페이지를 유지하고 이미지 평가를 수행합니다.

원본 WebUI HTML의 외부 CSS·폰트·이미지를 다시 불러와 정답을 바꾸지 않습니다. 수정 과정은
**생성된 HTML**을 실제 Chromium으로 실행합니다. reference AX 정답 박스는 평가에만 쓰며,
FrameDiff의 목표 박스는 다른 데이터셋과 동일하게 VLM이 screenshot에서 추정합니다.
목표 screenshot은 첫 번째 이미지, 현재 HTML에서 브라우저가 측정한 박스에 ID·역할·이름을 표시한
named frame은 두 번째 이미지로 입력합니다. named frame은 현재 위치만 표시하며 목표 위치나 Oracle
정보를 포함하지 않습니다.

WebUI에서는 `OFFICIAL_REPO`를 사용하지 않습니다(일괄 스크립트가 안내 후 무시).
이 점수들은 Design2Code 공식 지표나 WebUI 논문 지표를 재현한 것이 아닙니다.
`--initial-mode text-augmented`는 원본 HTML이 모든 사례에 있을 때만 지원하며 기본 direct를 권장합니다.

## 비교 조건

| Method | 공유 초기 HTML 이후 처리 |
|---|---|
| initial | 수정하지 않음 |
| self-revision-1 … N | 목표 screenshot + 현재 실제 페이지 screenshot + 현재 HTML을 VLM에 입력, 전체 HTML 수정 |
| coordinate-feedback | 같은 추정 박스 → 규칙 기반 단일 CSS 수정 후보 → 실제 HTML 재배치 → 후보 선택·반복 |
| model-feedback (기본) | 같은 추정 박스 → FrameDiff 단일 CSS 수정 후보 → 실제 HTML 재배치 → 후보 선택·반복 |
| coordinate / model (legacy) | 기존 proxy-only 탐색 후 마지막에만 HTML에 전체 박스 변화량 반영; ablation용 |

기본 feedback 루프는 `HTML → 실제 Chromium layout → 현재 박스 → 후보 → 단일 CSS 수정 →
전체 실제 layout 재계산 → 선택 → 반복`입니다. 중간에 전체 요소를 강제로 좌표에 맞추지 않습니다.
따라서 텍스트 줄바꿈·auto height·형제 이동의 연쇄 효과가 다음 모델 입력에 포함됩니다.

Self-Revision은 LayoutIR/추정 박스를 보지 않습니다. FrameDiff는 정답 HTML/DOM 박스를 보지 않습니다.
초기 생성, 목표 박스 추출, IR fitting, 반복 수정, 중간 브라우저 실행의 시간을 각 방법에 맞게 합산합니다.
모델 로딩과 사후 평가는 별도입니다. FrameDiff 학습 비용도 추론 시간에 포함되지 않습니다.
VLM 호출 수에는 실패한 요청도 포함됩니다. 중간 후보 생성 토큰·usage는 각 `*.meta.json`에 보존합니다.

이는 **시스템 수준의 품질/추론 비용 비교**입니다. 모델 크기·학습 데이터가 통제된 diffusion 학습법 비교는 아닙니다.

## Oracle target-box 2×2 진단

VLM 박스 오차와 수정 정책 오차를 분리하려면 Design2Code 정답 HTML의 동일 DOM을 사용합니다.
`web-oracle-prepare`는 정답 HTML을 한 번 렌더링해 stable ID와 Oracle 박스를 저장하고, 그 HTML에
`width`, `height`, `margin-left`, `margin-top` 교란을 순차 적용합니다. 정답과 교란 HTML은 요소 ID와
부모 관계가 같으므로 별도의 의미 기반 DOM matching 없이 박스 오차를 측정할 수 있습니다.
VLM 입력과 pixel 평가에는 이 정답 HTML을 동일 Chromium/viewport에서 렌더링한 screenshot을 사용해
데이터셋 원본과 로컬 렌더러의 폰트·외부 자산 차이가 박스 추정 오차에 섞이지 않게 합니다.
두 번째 VLM 이미지인 `current-named-frame.png`는 corrupted HTML의 정확한 현재 박스와 ID를 표시해
반복되는 텍스트와 유사한 컴포넌트의 identity 대응을 돕습니다. 목표 위치는 포함하지 않습니다.

먼저 5개 smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 CORRUPTIONS=4 VLM_RETRIES=2 \
bash scripts/run_oracle_ablation.sh \
  "$D2C_ROOT" runs/oracle-ablation-smoke runs/main/best.pt
```

실행되는 네 방법은 다음과 같습니다.

| Method | 목표 박스 | 후보 제안 |
|---|---|---|
| `coordinate-vlm` | 목표 screenshot + 현재 named frame에서 VLM이 추정 | 좌표 residual |
| `model-vlm` | 목표 screenshot + 현재 named frame에서 VLM이 추정 | FrameDiff checkpoint |
| `coordinate-oracle` | 정답 HTML의 실제 DOM 박스 | 좌표 residual |
| `model-oracle` | 정답 HTML의 실제 DOM 박스 | FrameDiff checkpoint |

모든 방법은 같은 corrupted HTML에서 시작하며 후보마다 Chromium으로 전체 DOM layout을 다시 계산합니다.
Oracle 정보는 `*-oracle` 진단에만 들어가며 end-to-end 성능 주장이 아닙니다. VLM 방법은 정답 HTML이나
Oracle 박스를 보지 않습니다. `prepare/prepare-report.json`은 VLM target box의 identity-aligned IoU,
center/size error를 기록합니다.

최종 파일:

- `evaluation/report.md`: 전체/공통 성공 집계와 2×2 분해 표
- `evaluation/oracle-ablation.json`: VLM 박스 정확도와 수치 분해
- `evaluation/paired-common-success.json`: 동일 성공 페이지의 initial 대비 변화
- `prepare/pages/*/corruption.json`: 요청한 교란과 실제 변경 박스
- `prepare/pages/*/current-named-frame.png`: VLM에 제공한 현재 ID 지도
- `repair/pages/*/*-steps/`: 각 방법의 실제 브라우저 trajectory

분해 표의 값은 모두 양수가 개선을 뜻하도록 방향을 통일합니다.
`Oracle target gain`은 같은 수정 정책에서 VLM 박스를 Oracle로 바꾼 효과이고,
`Model policy gain`은 같은 목표 박스에서 coordinate를 FrameDiff로 바꾼 효과입니다.
실패 페이지를 방법별로 다르게 제외하지 않도록 네 방법이 모두 성공한 공통 페이지에서 계산합니다.

전체 실행은 `PAGE_LIMIT`을 생략하고 새 출력 폴더를 사용합니다. `CORRUPTIONS`, `MAX_NODES`, `SEED`,
`FEEDBACK_RENDER`, `OFFICIAL_REPO`를 환경변수로 설정할 수 있습니다. 실제 논문 결과에는 여러 corruption
seed를 각각 실행해 평균과 분산을 보고하고, 이 controlled 결과와 기존 end-to-end 결과를 함께 제시해야 합니다.
named-frame 입력 이전 prepare 캐시는 protocol이 달라 재사용되지 않으므로 새 출력 폴더를 사용하세요.

## 설치

```bash
conda activate framediff-4090
python -m pip install -e '.[web,vlm]'
python -m playwright install --with-deps chromium
```

공식 Design2Code 지표도 실행하려면:

```bash
git clone https://github.com/NoviScl/Design2Code.git data/design2code-official
git -C data/design2code-official checkout 7a575e4c33f417c4be5c64072b8f5798de0d0f99
python -m pip install -e '.[visual-metrics]'
python -m pip install 'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
export OFFICIAL_REPO="$PWD/data/design2code-official"
```

가중치는 첫 평가 때 `.cache/clip`에 다운로드됩니다. 공식 repo의 Python 코드를 import하므로 신뢰할 수 있는
공식 checkout을 사용하세요. 데이터·가중치·외부 코드는 Git에 추가하지 않습니다.

## 한 번에 실행

먼저 별도 출력 디렉터리에서 5개 smoke run:

```bash
PAGE_LIMIT=5 REVISION_ROUNDS=1 bash scripts/run_real_web.sh \
  "$D2C_ROOT" runs/real-d2c-smoke runs/main/best.pt
```

전체 Design2Code와 Hard:

```bash
REVISION_ROUNDS=3 bash scripts/run_real_web.sh \
  "$D2C_ROOT" runs/real-d2c runs/main/best.pt

REVISION_ROUNDS=3 bash scripts/run_real_web.sh \
  "$D2C_HARD_ROOT" runs/real-d2c-hard runs/main/best.pt
```

`OFFICIAL_REPO`가 설정되면 공식 5개 지표까지 실행합니다. 미설정이면 Chromium DOM 기하 지표와 screenshot
pixel MAE만 계산합니다. 후자를 공식 Design2Code 점수로 부르면 안 됩니다.
세 프로세스가 순차 실행되므로 Qwen, FrameDiff, CLIP이 동시에 GPU를 점유하지 않습니다.
`CUDA_VISIBLE_DEVICES=1`을 앞에 붙이면 해당 GPU로 제한할 수 있습니다.

`REVISION_ROUNDS=3`은 1/2/3회 결과를 모두 저장합니다. 별도 VLM 후보 탐색이나 정답 기반 best-of-N 선택은 하지 않습니다.
중단 후 같은 명령으로 재개할 수 있습니다. 설정·데이터가 바뀌면 재개를 거부합니다.
따라서 smoke의 `PAGE_LIMIT=5`를 지우고 **같은 출력 폴더를 재사용하지 마세요**.
실패 결과도 캐시되므로 실패 페이지를 재시도할 때는 새 출력 폴더를 사용하세요.

API 서버 사용 예:

```bash
VLM_BACKEND=openai-compatible VLM_MODEL=my-served-vlm \
VLM_ENDPOINT=http://localhost:8000/v1/chat/completions PAGE_LIMIT=5 \
bash scripts/run_real_web.sh "$D2C_ROOT" runs/real-api-smoke runs/main/best.pt
```

인증은 `VLM_API_KEY` 환경 변수로 전달합니다. 원격 API를 지정하면 목표 이미지와 생성 코드가 해당 서버에 전송됩니다.

## 단계별 실행 / 기존 초기 HTML 사용

```bash
python -m framediff web-prepare \
  --root "$D2C_ROOT" --out runs/real-d2c/prepare \
  --backend qwen --four-bit --rounds 3 --resume

python -m framediff web-repair \
  --data runs/real-d2c/prepare/prepared.jsonl \
  --checkpoint runs/main/best.pt --out runs/real-d2c/repair \
  --device cuda --methods coordinate-feedback,model-feedback --feedback-render frames \
  --steps 10 --beam 2 --topk 8 --budget 160 --resume

python -m framediff web-evaluate \
  --data runs/real-d2c/repair/results.jsonl --out runs/real-d2c/evaluation \
  --official-repo data/design2code-official --resume
```

`--root` 대신 `--manifest input.jsonl`을 지정할 수 있습니다. 한 줄 예:

```json
{"id":"page-1","group":"page-1","screenshot":"/datasets/test/1.png","html":"/datasets/test/1.html","initial_html":"/predictions/1.html"}
```

`html`은 **평가 전용 reference**, `initial_html`은 **실제 수정 대상**입니다. 두 필드를 혼동하지 마세요.
`initial_html`이 있으면 초기 VLM 생성을 생략합니다. 기존 `prepare-design2code` manifest도 사용할 수 있으나,
그 안의 reference boxes는 이 파이프라인의 수정 입력으로 사용하지 않습니다.

원 논문은 text-augmented 초기 결과를 Self-Revision 입력으로 사용합니다. 이 조건은
`--initial-mode text-augmented` 또는 `INITIAL_MODE=text-augmented`로 선택합니다.
이때만 reference HTML에서 추출한 **텍스트**가 공통 초기 생성에 제공됩니다(박스·CSS는 제공하지 않음).
기본값 direct는 screenshot만 사용하는 변형이며 논문의 정확한 재현으로 표시하면 안 됩니다.

공식 prompt와 입력 구성을 포함한 1회 Self-Revision은 아래처럼 명시합니다.

```bash
python -m framediff web-prepare --root "$D2C_ROOT" --out runs/d2c-paper/prepare \
  --repair-conditioning visual --initial-mode text-augmented --rounds 1 \
  --revision-protocol design2code --seed 2024 --max-new-tokens 4096 --vlm-retries 0
```

이때 `initial_html`을 받지 않고 text-augmented initial부터 새로 생성하며 결과 method 이름은
`design2code-self-revision`입니다. 정답 HTML의 텍스트만 사용한다는 사실은
`uses_reference_text=true`로 기록됩니다. 목표 layout box나 정답 CSS는 입력하지 않습니다.

## 결과 읽기

- `evaluation/report.md`: 방법별 결과 및 공식 지표 표.
- `evaluation/summary.json`: 평균, 지표별 평가 성공 수 `*_n`, 수정 실패율과 평가 실패 수.
- `evaluation/paired-vs-initial.json`: 동일 페이지 초기 결과 대비 변화량·개선 비율.
- `evaluation/metrics.jsonl`: 페이지별 지표, 시간, VLM/브라우저 호출 수, 실패 원인.
- `prepare/pages/*`: 초기 HTML, 단계별 Self-Revision HTML, 이미지·프롬프트·원문·메타데이터, IR fitting 보고.
- `repair/pages/*`: 실제 수정 HTML, IR와 탐색 trace.
- `repair/pages/*/*-feedback-steps/step-NNN.*`: 선택된 중간 HTML, 실제 박스 JSON, 이름 붙은 frame PNG.
  모든 후보의 score와 영향을 받은 요소 ID는 `*-trace.json`의 `candidate_log`에 있습니다.
- `evaluation/pages/*`: 동일 viewport의 최종 실제 screenshot, reference 재렌더링, 공식 평가 provenance.

`pipeline_seconds`는 최종 HTML을 얻기까지의 시간입니다. `feedback_image_seconds`, `feedback_images`,
`feedback_browser_screenshots`, `candidate_failure_count`로 중간 이미지 생성·실패 비용을 확인합니다.
`evaluation_seconds`와
`evaluation_browser_executions`는 독립적인 사후 검증 비용입니다. 수정 실패는 마지막 유효 HTML로 평가하며
페이지를 제외하지 않습니다. 평가 자체가 실패하면 null/누락으로 기록하고 `*_n`을 확인해야 합니다.
누락된 공식 점수를 0이나 성공으로 대체하지 않습니다.

## 실제 HTML 중간 피드백의 범위와 한계

1. 초기 실제 DOM에서 `max_nodes-1`개까지 요소를 선택하고 고유 ID를 붙입니다. 전체/선택 노드 수를 기록합니다.
2. DOM 박스를 LayoutIR로 근사합니다. 이 과정의 IoU와 reparenting 수를 저장하며, 낮은 fitting 품질 때문에
   평가 페이지를 임의로 제외하지 않습니다. 기존 IR는 높이 1024px 등의 표현 한계가 있어 긴 페이지에서 오차가 클 수 있습니다.
3. 기본 `*-feedback` 경로는 모델의 후보를 **하나의 요소·하나의 CSS 속성 수정**으로 바꿉니다.
   width/height는 기존 box-sizing을 고려해 px 크기를 수정하고, dx/dy는 margin-left/margin-top 변화로 적용합니다.
   다른 요소의 width/height를 고정하지 않으며, width만 바꾸면 기존 auto height는 그대로 남습니다.
   flow/grid 구조 변경 등은 아직 허용하지 않고 후보 마스크로 제외합니다.
4. 각 후보를 부모 상태의 실제 HTML에 적용하고 Chromium으로 모든 추적 요소의 박스를 다시 읽습니다.
   IDs는 다시 매기지 않습니다. 숨겨지거나 크기가 0이 된 요소도 추적하며, ID 소실은 실패로 기록합니다.
   후보 점수와 다음 모델 입력 모두 실제 박스입니다. 모델용 양자화 특징은 실제 측정값으로 갱신하며
   IR ancestry는 고정합니다. CSS min/max 크기·table·transform이 의도한 변경을 막아도 강제로 다른 요소를 옮기지 않습니다.
5. 최종 점수는 **수정된 실제 HTML 렌더링**에서 구합니다. 선택되지 않은 요소·텍스트 줄바꿈도
   최종 이미지 평가에 포함됩니다. 탐색 중 점수는 추적된 요소/추정 목표 박스만 사용하므로, 추정 오류와
   노드 선택의 한계는 남습니다. HTML/CSS 변경의 연쇄 효과를 관찰한다는 것이 모든 오류를 해결한다는 뜻은 아닙니다.

기존 `coordinate`/`model`은 이전의 proxy-only + 전체 좌표 변화량 최종 적용 경로입니다.
그 경로의 `transfer_max_error_px`는 최종 목표 좌표와 실제 좌표 차이입니다. 새 feedback 경로는
전체 좌표 투영을 하지 않으므로 해당 필드가 null인 것이 정상입니다.
두 경로는 CSS action/적용 방식도 다르므로, 차이를 중간 피드백 **하나만의 효과**로 해석하면 안 됩니다.

## 간소화 프레임 vs 전체 screenshot 비용

`--feedback-render` 또는 일괄 스크립트의 `FEEDBACK_RENDER`로 선택합니다.

| 모드 | 실제 HTML layout | 추가 이미지 생성 |
|---|---|---|
| boxes | 모든 후보에서 수행 | 없음 |
| frames (기본) | 모든 후보에서 수행 | 측정 박스와 이름으로 작은 규칙 기반 frame 이미지 생성 |
| raster | 모든 후보에서 수행 | 실제 전체 viewport screenshot 생성 |

**세 모드 모두 기존 FrameDiff 모델에는 실제 박스·이름·트리 특징을 입력합니다.** 프레임/페이지 이미지를
이미지 인코더로 입력하는 새 모델을 학습시킨 것은 아닙니다. 이미지들은 확인·비용 비교용이며, 따라서 이 세 모드의
정확도가 비슷한 것은 자연스럽습니다. 브라우저 레이아웃 비용을 제거했다는 주장도 아닙니다.
중간 모든 후보에서 선택한 종류의 이미지를 생성하되 디스크에는 단계별 선택된 상태만 저장합니다.

기존 prepare 결과를 재사용하고 모드별 별도 출력 폴더에서 비교:

```bash
for render_mode in boxes frames raster; do
  python -m framediff web-repair \
    --data runs/real-d2c/prepare/prepared.jsonl --checkpoint runs/main/best.pt \
    --out "runs/real-d2c-feedback-$render_mode/repair" --device cuda \
    --methods coordinate-feedback,model-feedback --feedback-render "$render_mode" \
    --steps 10 --beam 2 --topk 8 --budget 160 --resume
  python -m framediff web-evaluate \
    --data "runs/real-d2c-feedback-$render_mode/repair/results.jsonl" \
    --out "runs/real-d2c-feedback-$render_mode/evaluation" --resume
done
```

기존 proxy 경로도 함께 비교하려면 `--methods coordinate,model,coordinate-feedback,model-feedback`를 사용하세요.
기존 repair 캐시는 새 protocol과 다르므로 새 출력 폴더를 사용해야 합니다.
기존 synthetic checkpoint로 실행은 가능하지만 실제 CSS reflow 분포로 학습된 것은 아니며, 성능 우위를 보장하지 않습니다.
저장된 실제 브라우저 trajectory는 후속 데이터 수집/분석용이며, 자동 재학습을 수행하지 않습니다.

한 screenshot 크기를 viewport로 사용하며 responsive/다중 viewport 성능을 주장하지 않습니다.
JS/외부 네트워크는 모든 방법에서 비활성화합니다. reference 옆의 `rick.jpg`만 명시적으로 읽어 이미지에 내장하며,
다른 외부 자산/폰트는 차단됩니다. 원본 screenshot과 재렌더링은 viewport·폰트 때문에 차이가 날 수 있습니다.

## VLM 출력 오류 재시도와 실패 결과 해석

`web-prepare --vlm-retries 2`는 각 HTML/목표 박스 요청에 최대 2회 추가 검증 재시도를 합니다.
잘린 HTML은 짧고 완결된 HTML을 다시 요청하고, JSON 문법 오류나 viewport/ID 오류는 오류 내용과
누락/추가 ID를 전달해 전체 응답을 다시 요청합니다. 박스는 임의로 채우지 않습니다.
CUDA OOM 같은 실행 오류는 재시도하지 않습니다. 각 시도의 prompt/raw/meta를 보존하며 모든 호출과
소요 시간을 집계합니다. Self-Revision의 검증 재시도는 새 revision round가 아닙니다.

```bash
CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 REVISION_ROUNDS=1 VLM_RETRIES=2 \
bash scripts/run_real_web.sh "$D2C_ROOT" runs/web-feedback-validated runs/main/best.pt
```

스크립트에서 `MAX_NEW_TOKENS`(기본 16384), `MAX_PIXELS`(기본 1048576)도 지정할 수 있습니다.
기존 버전 실행과 설정이 달라졌으므로 첫 실행은 새 출력 폴더를 사용하세요.
이 버전에서 만든 prepare 실행은 동일 설정의 `--resume --retry-failed`로 실패 페이지 전체를 재생성할 수 있습니다.
성공 페이지는 재사용하고, 실패 페이지의 이전 파일은 `previous-attempts` 아래 보관합니다.
준비 결과가 변경되면 repair/evaluation은 **새 출력 폴더**를 사용해야 합니다(입력 해시 검증).

`summary.json`은 실패 시 유지된 HTML을 포함한 전체 결과입니다.
`summary-common-success.json`과 `paired-common-success.json`은 모든 방법이 성공한 동일 페이지 집합을
사용하며 `page_ids`/`n` 또는 지표별 `*_n`을 확인해야 합니다. 공통 성공 집합은 성공 조건부 결과라서
전체 데이터셋 성능을 대신하지 않습니다. `report.md`에서 두 집계를 함께 보여줍니다.

`prepare/prepare-summary.json`은 선택된 원래 test 표본 수, 준비 단계별 실패 수와 상위 사유를 보존합니다.
최종 `evaluation/coverage.json`과 `evaluation/report.md`는 이 원래 분모를 이어 받아 준비 탈락/실패,
방법별 rollout 실패, metric 계산 실패와 end-to-end 성공률을 함께 보고합니다. 반복 실험은 trial 수를
페이지 수로 오인하지 않도록 페이지별로 묶고, 한 페이지의 반복 중 하나라도 실패하면 그 페이지를
end-to-end 성공으로 세지 않습니다. 품질 지표는 측정 가능한 표본에서 계산하며, 탈락 표본을 0점으로
바꾸지 않습니다.

## 공식 지표 연동 범위 (상세)

[공식 코드](https://github.com/NoviScl/Design2Code)의 `visual_eval_v3_multi`와 OCR-free block extraction,
matching/merging, CIEDE2000, CLIP-ViT-B/32를 직접 사용합니다. 5개 지표는
`official_block`, `official_text`, `official_position`, `official_color`, `official_clip`입니다.

상류 코드가 파일을 변경하고 shell을 실행하기 때문에 **임시 복사본**에만 preprocessing을 적용하고,
shell 호출은 실행하지 않고 같은 network-disabled Chromium 렌더러로 교체합니다.
알려진 upstream 빈 block 분기의 indentation 오류만 메모리에서 교정하고 provenance에 기록합니다.
공식 소스 SHA를 결과에 남깁니다. metric 수식은 상류 그대로지만 viewport·렌더링 환경이 다르므로
**논문 표의 숫자를 재현했다는 의미는 아닙니다**. 이 실행에서 얻은 방법 간 paired 비교에 사용하세요.

공식 연동 테스트(선택 의존성과 가중치 필요):

```bash
FRAMEDIFF_OFFICIAL_REPO="$OFFICIAL_REPO" python -m pytest tests/test_web_experiment.py -k official -q
```
