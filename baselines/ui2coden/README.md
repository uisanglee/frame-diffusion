# UI2CodeN 비교 실험 — 독립 실행 폴더

공식 [UI2CodeN](https://github.com/zai-org/UI2Code_N)의
[`zai-org/UI2Code_N`](https://huggingface.co/zai-org/UI2Code_N) 공개 가중치와
[Transformers 데모](https://github.com/zai-org/UI2Code_N/blob/main/demo/demo_local.py)의
입력 방식을 이용한 **우리 평가 프로토콜용 polishing 실행기**다.
학습/재학습이 아니라 추론 실험이며, 기존 `framediff/`, `scripts/`, 데이터,
체크포인트는 수정하지 않는다. 별도 `.venv`에만 의존성을 설치한다.

## 실험의 정의

각 페이지에 대해 **TUIDE와 같은 초기 HTML**, 같은 목표 스크린샷과 viewport를 사용한다.

```
목표 PNG + 현재 HTML + 현재 HTML의 렌더링 PNG
    → UI2CodeN → 수정된 전체 HTML → 렌더링 → 다음 round
```

기본 3회 수정이며 초기 상태와 round 1/2/3을 모두 평가한다. `ROUNDS`로 변경한다.
목표 HTML·정답 박스는 모델에 전달하지 않고 최종 평가에서만 사용한다.
점수로 후보를 선택하거나 악화 시 rollback하지 않는다. 잘린 HTML, 실행/렌더링 오류, 화면 보존 검증에 실패한 후보를
실패로 기록하고 이전 HTML을 유지한다. 실패 페이지도 평가에 포함한다.

`PROTOCOL=simple`은 목표 이미지와 현재 렌더링을 비교해 HTML을 수정하고,
완전하고 실행 가능한 HTML 문서를 반환하도록 요청한다. UI2CodeN에는 React/JSX/Babel 및
Tailwind CDN 금지 지시를 넣지 않는다. 별도 애플리케이션 렌더러에서 JavaScript를 실행하고,
허용된 라이브러리 의존성을 캐시해 로딩한다. 학습용 정적 렌더러 설정은 변경하지 않는다.
데이터셋별 이미지 규칙을 추가한다. Design2Code/Hard는 `rick.jpg`, 색상 치환 WebUI는
빨강/초록/노랑/파랑/검정에 각각 `rick1.jpg`~`rick5.jpg`를 사용하도록 안내한다.
기존 콘텐츠·자원·앱 동작을 유지하며 레이아웃/크기/간격 개선에 집중하고,
새 이미지·SVG 그림·canvas·base64 이미지와 추가 이미지/아이콘/폰트 팩은 만들지 않도록 요청한다.
이 지시는 프롬프트이며 모델 출력의 완전한 준수를 보장하지 않는다. React/Tailwind 실행은 유지한다.
프롬프트 변경 전 결과와 섞이지 않도록 새 출력 디렉터리를 사용한다.
공통 문구는 `framediff/ui2code_asset_prompt.py`, 실행기 문구와 contract는 `runner.py` 및
출력 `config.json`에 기록된다.

각 후보의 원본 애플리케이션은 `application-N/application.html`, 실행된 DOM/CSS는 capture
메타데이터가 가리키는 정적 파일로 보존한다. 실행 화면과 정적 캡처 화면의 픽셀·박스가
허용 오차 안에서 일치해야 후보를 채택한다. 평가용 `output-N.html`에는 검증된 정적 캡처를
저장하지만, 다음 polishing 프롬프트에는 `application_html`의 원본 JavaScript/JSX를 전달한다.
후보 실패 시 마지막 정상 정적 HTML과 원본 애플리케이션 쌍을 모두 유지한다.
실행·캡처·검증 시간은 해당 회차 비용에 포함되며, `application_capture`와
`application_attempt`에 결과 및 비용을 기록한다. 이전 정적-only 실행 결과와 혼합하지 않는다.

이미지·폰트 등의 바이너리는 `TUIDE_LOCAL_ASSET_CACHE`(기본 `.cache/application-assets`)의
해시 기반 파일로 중복 없이 저장한다. HTML/CSS와 다음 round의 프롬프트에는 base64 대신 짧은
`https://framediff-assets.invalid/...` 주소만 남긴다. 이 주소는 인터넷으로 요청하지 않고
프로젝트 렌더러가 로컬 파일을 검증하여 제공한다. HTML을 평가용 임시 디렉터리로 복사해도
그대로 로딩되며, 누락·변조 자원은 실패 처리한다. 외부 CSS의 기존 자원도 원본 바이트를 보존한다.
`freeze.json`의 `local_asset_cache`, `input_html_chars`, `source_html_chars`, `static_html_chars`,
`pixel_mae`, `max_box_error_px`로 용량과 화면 보존을 확인할 수 있다.
서버를 옮기면 이 캐시와 `TUIDE_APP_ASSET_CACHE`(기본 `.cache/app-dependencies`)도 함께 복사한다.
일반 브라우저로 HTML만 열면 전용 자원 주소를 로딩할 수 없으므로 프로젝트 렌더러를 사용한다.

이 구현은 공식 논문 Table 1의 정확한 재현을 주장하지 않는다. 공개 데모를 이용해
반복 루프를 구성했고, 반복 횟수·간소화 프롬프트·양자화·입력 축소는 이 실험의 설정이다.
논문에는 실제 적용한 정밀도(BF16/NF4), 반복 횟수, 프롬프트 및 애플리케이션 실행·캡처
프로토콜을 명시한다.

## 4090 설치 — 한 번만

Linux, Python 3.11/3.12, 동작하는 NVIDIA 드라이버를 전제로 한다.
설치기는 PyTorch 2.8.0 / torchvision 0.23.0의 CUDA 12.8 휠을 고정하여
bitsandbytes 0.48과 호환되게 설치하고, 실제 GPU NF4 연산을 검사한다.
기존 전용 환경에도 다시 실행할 수 있으며 학습 환경과 모델 다운로드 캐시는 유지한다.
`BNB_CUDA_VERSION`을 따로 설정했다면 해제하고 실행한다.

```bash
cd /home/uisang/2027/frame-diffusion
bash baselines/ui2coden/setup_4090.sh
```

Chromium 시스템 라이브러리가 없다는 오류가 나면 서버 관리자가 다음을 실행한다.
기존 학습 환경에는 패키지를 설치하지 않는다.

```bash
sudo baselines/ui2coden/.venv/bin/python -m playwright install-deps chromium
```

기본값: **GPU 한 장, NF4 4비트, BF16 계산, SDPA, batch 1, CPU 2 threads**.
목표/현재 이미지는 각각 최대 262,144 pixels로 축소해 모델에 넣는다.
평가 렌더링은 원래 viewport 해상도다. 단독 실행의 기본 출력 상한은 16,384 tokens이며,
논문 실행기는 WebUI에 16,384, Design2Code/Hard에 4,096을 전달한다.
HTML 완성 또는 EOS에서 먼저 종료할 수 있다.
입력+출력 한도는 고정된 모델 revision의 `text_config.max_position_embeddings`를 읽는다.
`MAX_CONTEXT=0`(기본)은 이 모델 한도를 사용하며 양수를 지정하면 더 낮은 제한을 적용한다.
입력 17,000 + 출력 예산 16,384는 모델 문맥 한도가 33,384 이상일 때 허용된다.
실제 입력에는 이미지 토큰·지시문·현재 HTML이 모두 포함된다. 입력이나 출력 예산을
자동으로 줄이지 않는다. 메모리 확보는 별개이며 긴 입력의 4090 실행을 보장하지 않는다.
NF4, batch 1, SDPA와 `logits_to_keep=1`을 사용한다. OOM은 즉시 중단한다.
`runtime.json`에 문맥 한도와 출력 예산, 각 round의 `generation`에 실제 토큰 수를 기록한다.
논문 실행기는 데이터셋에 맞춰 `UI2CODEN_MAX_NEW_TOKENS`를 Initial/Self Revision의
`MAX_NEW_TOKENS`와 동일하게 설정하고, `UI2CODEN_MAX_CONTEXT=0`을 사용한다.
긴 HTML의 실패율도 함께 보고해야 한다. 데이터 크기에 따라 이 제한을 조정하려면
별도 출력 폴더를 사용한다. 단일 GPU 모드는 여유 VRAM이 18 GiB 미만이면 실행을 거절한다.
이 기준은 사전 자원 점검이지 OOM 방지 보장은 아니다.

### 양자화 없이 BF16, GPU 두 장

`VLM_DEVICE_MAP=balanced UI2CODEN_PRECISION=bf16 GPU=0,1`을 지정하면 동일한
polishing 루프에서 모델 레이어를 두 GPU에 분산한다. `VLM_GPU_MEMORY_GIB=20`은
GPU당 **가중치 배치 예산**이며 실행 중 VRAM의 하드 상한이 아니다. 실제 여유 메모리에서
최소 2 GiB를 제외한 값과 이 예산 중 작은 값으로 배치를 계획한다. 남은 VRAM은
KV 캐시와 이미지 처리에 쓰인다. 가중치가 두 GPU에 들어가지 않으면 로딩 전에
실패하며 CPU/disk로 자동 offload하지 않는다. 긴 문맥의 OOM 방지를 보장하지 않는다.

```bash
GPU=0,1 VLM_DEVICE_MAP=balanced UI2CODEN_PRECISION=bf16 \
  VLM_GPU_MEMORY_GIB=20 ROUNDS=2 \
  bash baselines/ui2coden/run_4090.sh INPUT.jsonl NEW_OUTPUT_DIR
```

`INPUT.jsonl`은 기존 prepared manifest, `NEW_OUTPUT_DIR`은 새 출력 디렉터리로 바꾼다.
기존 4bit 결과와 같은 디렉터리에서 재개하지 않는다. 두 GPU에 레이어를 나누는
방식은 처리량을 두 배로 만드는 tensor parallelism이 아니며 속도는 실측해야 한다.
SDPA, KV 캐시, 지원되는 모델의 마지막 토큰 logits 계산을 사용한다.
`runtime.json`에는 정밀도·실제 device map·GPU 정보가 저장되고 round별 `generation`에는
각 GPU의 피크 메모리와 생성 시간이 저장된다. 메모리는 PyTorch allocated 기준이다.

모델 최초 다운로드에는 네트워크와 디스크 공간이 필요하다. HF 캐시는 기존
`HF_HOME` 설정을 따른다. 학습 캐시는 만들지 않는다. 최초 모델 revision을 SHA로
저장하고 재개 때 같은 SHA를 사용한다. 설치 버전은 `.venv/installed-requirements.txt`,
모델·GPU·Torch 버전은 출력 `runtime.json`에 저장한다.

## 입력

다음 세 형식을 읽는다. **한 실험 안에서는 동일한 데이터 출처/프로토콜을 유지한다.**

1. 기존 외부 UI 평가 manifest: `id`, `screenshot`, `initial_html`, `viewport`, 선택 `html`.
2. 기존 prepared manifest: 위 필드 중 초기 HTML은 `methods.initial.html`에서 읽는다.
3. 고정 corruption `policy-test.jsonl`: `id`, `target_image`, `current_html`,
   `target_html`, `viewport`. train이나 val 대신 **test**를 지정한다.

최소 JSONL 한 줄 예시 (절대 경로 권장):

```json
{"id":"page-001","page_id":"page-001","screenshot":"/data/target.png","initial_html":"/data/initial.html","html":"/data/reference.html","viewport":[1280,720]}
```

상대 경로는 프로젝트 루트 기준이다. 다른 기준은 직접 `runner.py --path-root ...`로 지정한다.
`page_id`가 있으면 같은 페이지의 반복 측정은 기존 보고서에서 묶인다.
HTML이 없으면 pixel MAE는 측정하지만 정답 DOM 기하 지표는 계산할 수 없다.
`reference_kind=webui_recorded_boxes`와 `reference_boxes`도 기존 평가기에 전달한다.

참조 이미지와 점수 계산에는 기존의 **JavaScript·네트워크 차단 정적 렌더러**를 사용한다.
UI2CodeN 후보만 별도 렌더러에서 실행한 후 화면 보존을 검증한 DOM/CSS 캡처로 평가한다.
React/ReactDOM/Babel/Tailwind 등 허용된 CDN 의존성은 다운로드·캐시하지만,
임의 외부 API와 이미지 다운로드, 별도 npm 프로젝트 빌드는 지원하지 않는다.
`rick.jpg` 및 WebUI 색상 이미지 같은 제공된 자산은 data URI로 포함한다.
React 원본은 `application_html`로 보존하고 정적 캡처는 `html`로 구분한다.

## 먼저 2페이지 검증

```bash
cd /home/uisang/2027/frame-diffusion
GPU=2 LIMIT=2 ROUNDS=1 bash baselines/ui2coden/run_4090.sh \
  data/webui-full-dom-screened-7k-v1/7000/css/policy-test.jsonl \
  baselines/ui2coden/runs/7k-smoke
```

가중치를 로드하지 않고 입력 경로만 확인하려면:

```bash
baselines/ui2coden/.venv/bin/python baselines/ui2coden/runner.py \
  --manifest data/webui-full-dom-screened-7k-v1/7000/css/policy-test.jsonl \
  --out baselines/ui2coden/runs/unused --check-inputs
```

## 전체 test 실행 — GPU 2, 로그 저장

```bash
cd /home/uisang/2027/frame-diffusion
mkdir -p baselines/ui2coden/runs
nohup env GPU=2 ROUNDS=3 \
  bash baselines/ui2coden/run_4090.sh \
    data/webui-full-dom-screened-7k-v1/7000/css/policy-test.jsonl \
    baselines/ui2coden/runs/7k-layout-n3 \
  > baselines/ui2coden/runs/7k-layout-n3.log 2>&1 < /dev/null &
```

외부 VLM 초안과 비교할 때는 첫 번째 경로를 **TUIDE에 사용한 동일한 prepared
manifest**로 바꾸고 새 출력 폴더를 지정한다. 위 고정 corruption 실험과 섞지 않는다.
같은 명령을 재실행하면 완료된 round는 재생성하지 않는다. 같은 출력 폴더의 동시 실행은
lock으로 막는다. 입력/설정/실행 코드가 변경되면 재개를 거절한다.
CUDA 오류는 다음 샘플로 계속 진행하지 않고 프로세스를 종료한다.

일반 종료는 실행 중인 runner의 PID를 확인해 `kill -TERM PID`를 사용한다.
runner는 종료 신호에서 Chromium context를 닫으며, `kill -9`은 정리를 생략한다.
쉘의 `set -e` 때문에 실패한 추론 뒤 평가가 시작되지 않는다.

## 결과 및 지표

`run_4090.sh`는 모델 프로세스가 종료되어 VRAM을 반환한 다음 기존
`python -m framediff web-evaluate`를 실행한다. GPU를 쓰는 CLIP 평가와 VLM을
동시에 올리지 않는다.

| 출력 | 내용 |
|---|---|
| `results.jsonl` | 초기 상태와 모든 round, 기존 평가기 호환 manifest |
| `pages/*/input-N.png` | 모델이 비교할 현재 렌더링 (파일은 원래 해상도) |
| `pages/*/answer-N.txt` | 원문 모델 출력 |
| `pages/*/output-N.html`, `output-N.png` | 유효한 수정 결과 |
| `pages/*/round-N.json` | 재개 상태, 누적 시간·호출·토큰, 실패, 최대 CUDA 메모리 |
| `evaluation/metrics.jsonl` | 페이지별 품질/비용/실패 지표 |
| `evaluation/paper-report.md`, `paper-summary.csv` | 기존 논문용 집계와 CI |

기본 지표: **DOM box IoU, center/size error, pixel MAE**, 기존 평가기의
element/page success 및 relation/overflow 진단. `webui_recorded_boxes` 입력은
`webui_*`, 정답 HTML 렌더링 입력은 `dom_*` 기하 지표가 나온다.
시간은 각 round까지 누적된 `repair_seconds`, 호출/토큰은 `repair_*`를 비교한다.
모델 다운로드/최초 로드는 repair 시간에서 제외한다. 초기 생성 비용이 manifest에
없으면 초기 비용은 0으로 기록되므로 **총 VLM drafting 비용으로 해석하면 안 된다**.

여기서 기하 대응은 기존 **web-evaluate의 Hungarian matching**이다.
고정 corruption TUIDE의 `visual-test-rollout`은 보존된 DOM ID를 사용하는 지표도
있으므로 그 수치를 그대로 같은 표에 섞지 않는다. 전체 HTML을 다시 쓰는 VLM은
DOM ID를 유지하지 않을 수 있다. 공정한 방법 비교에는 TUIDE 출력도 동일한
`web-evaluate`, 같은 `MAX_NODES`(기본 2048), 같은 viewport로 평가한다.

### 선택: 공식 Design2Code Block / Position / CLIP

기존 프로젝트가 사용하는 공식 지표 어댑터를 그대로 호출한다. 별도로 준비한
Design2Code checkout 경로를 넣는다. 정답 HTML이 필수이며 recorded-box 모드에는
사용하지 않는다. 전체 데이터 실행 전에 작은 샘플로 의존성 호환성을 확인한다.

```bash
baselines/ui2coden/.venv/bin/python -m pip install -r baselines/ui2coden/requirements-metrics.txt
GPU=2 ROUNDS=3 OFFICIAL_REPO=/absolute/path/to/Design2Code \
  bash baselines/ui2coden/run_4090.sh /absolute/path/to/shared-manifest.jsonl \
  baselines/ui2coden/runs/external-layout-n3
```

추론을 다시 하지 않고 나중에 공식 지표만 추가할 수도 있다. 기존 평가 폴더와 다른
`--out`을 사용한다.

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$PWD" baselines/ui2coden/.venv/bin/python \
  -m framediff web-evaluate \
  --data baselines/ui2coden/runs/external-layout-n3/results.jsonl \
  --out baselines/ui2coden/runs/external-layout-n3/evaluation-official \
  --official-repo /absolute/path/to/Design2Code --max-nodes 2048 --resume
```

## 검증 범위

```bash
baselines/ui2coden/.venv/bin/python -m pytest -q baselines/ui2coden/tests
```

테스트는 round 연결/누적 비용, 재개, 출력 변조 검출, 잘린 응답 fallback,
CUDA 오류 중단, policy manifest 변환, **실제 Chromium → 공통 평가 → 논문 보고서**를
검증한다. 가짜 모델 출력을 사용하는 기능 테스트이며, 로컬 Mac에서는 실제 4090의
9B 추론·처리량·정확도 및 선택적 공식 CLIP 실행을 검증하지 않았다.
서버의 2페이지 smoke 결과를 확인한 뒤 전체 실험을 시작한다.

## Polishing 생성 설정

Polishing은 고정한 `zai-org/UI2Code_N` revision의 `generation_config.json`을
명시적으로 로드한다. `generate()`에는 입력과 데이터셋별 `max_new_tokens`
(WebUI 16,384, Design2Code/Hard 4,096),
완성된 HTML 문서에서 종료하는 `stopping_criteria`를 전달하며,
`do_sample=False` 등 생성 설정을 덮어쓰지 않는다. 공식 공개 설정은
`do_sample=true`, `temperature=0.8`, `top_p=0.6`, `top_k=2`이다.
EOS/padding도 해당 revision의 값을 사용한다. 설정 파일 로딩 실패는 오류로 처리한다.
HTML 문서가 완성되면 출력 상한 전에 종료하고, reasoning·스크립트·주석 안의
`</html>` 문자열에는 종료하지 않는다. 각 round의 `generation_stop_reason`에
문서 완성·EOS·출력 상한 중 종료 사유를 기록한다.
출처: https://huggingface.co/zai-org/UI2Code_N/blob/main/generation_config.json

`runtime.json`과 각 round JSON의 `generation.generation_settings`에 실제 설정을
기록한다. 이전 greedy 결과는 설정/runner 변경 검사로 재개를 거부하므로 새 출력
디렉터리를 사용해야 한다. 기존 결과를 자동 삭제하거나 다시 실행하지 않는다.
이는 **생성 설정**의 일치이며, 기존 실험용 프롬프트·이미지 전처리·평가 프로토콜까지
공식 실험과 동일하다는 의미는 아니다.
