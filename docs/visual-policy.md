# Image-conditioned FrameDiff

이 경로는 **고정 계획/목표 좌표 residual 없이 이미지로 action을 예측**한다.
기존 box/plan 체크포인트와 호환되지 않으며 새로 학습해야 한다.
반복 역편집(denoising) 정책이며 Gaussian DDPM은 아니다. v2 action contract는 한 step을
하나의 문법적으로 허용된 CSS declaration 변경으로 제한하는 discrete structured diffusion이다.

## 모델과 입력

1. 추상화 모델: ImageNet 사전학습 ResNet-50 + FPN + Faster R-CNN 검출 head.
   목표 screenshot → text/image/control/painted-region box·class → 색상 사각형 이미지.
   이번 구현은 검출 head이며 별도 segmentation head는 없다.
2. 이미지 조건 정책: ImageNet 사전학습 ResNet-18 + 다중 해상도 특징(stride 4/8/16),
   spatial tokens, 현재 DOM ROIAlign, target/current cross-attention, tree attention.
   이미지 픽셀과 현재 DOM 구조·속성·box를 입력한다. 목표 box/정답 HTML/계획은 입력하지 않는다.
3. 행동: node 선택 + CSS declaration 하나의 변경 또는 STOP. 수치 변화는 해당 viewport 축의
   ±0.3125/0.625/1.25/2.5/5%로 제한한다. 지원 grammar는 width/height, 4방향 margin과 padding,
   row/column gap, flex-direction, justify-content, align-items다. 실제 CSS 재실행 후 줄바꿈·자동
   높이·형제 이동을 다음 스텝에 반영한다.

모든 action에 `sigma_decl(a)=변경 declaration 수=1`을 적용한다. 수치형에는 추가로
`sigma_mag(a)=abs(delta_px)/viewport_axis <= 0.05`를 적용한다. action은 여러 declaration이나
여러 DOM node를 동시에 수정할 수 없다. CSS reflow로 여러 자식 box가 바뀌는 것은 action의 실행
결과이지 추가 action이 아니다.

| 방식 | 목표 입력 | 매 스텝 현재 입력 |
|---|---|---|
| screenshot-policy | 목표 screenshot 특징을 한 번 캐시 | Chromium screenshot → image encoder |
| abstract-policy | 목표 screenshot → 검출기 한 번 → 추상화 이미지·특징 캐시 | Chromium DOM box → 추상화 이미지 → image encoder |
| abstract-oracle (선택) | 정답 DOM 추상화 이미지 | 위와 동일; controlled 데이터에서만 허용 |

abstract-policy도 **브라우저 layout 실행은 필요**하다. 생략하는 것은 매 스텝 full screenshot
캡처와 고해상도 시각 처리다. 두 정책 모두 목표 특징을 캐시한다.
목표와 현재의 ID 일치를 가정하지 않으며 추상화 이미지에 ID는 없다.
목표 IoU로 후보를 선택하는 검색은 없다. greedy action이므로 매 스텝 개선은 보장하지 않는다.

## 설치·전체 학습

저장소 루트에서 실행한다. 기존 환경이면 `pip install -e '.[visual,test]'`로 갱신한다.

```bash
conda env create -f environment-4090.yml
conda activate framediff-4090
python -m playwright install chromium

CUDA_VISIBLE_DEVICES=0 \
bash scripts/train_visual.sh data/visual runs/visual
```

기본 순서: procedural HTML 1,000개 → 실제 브라우저 학습 데이터 → 검출기 10,000 steps →
held-out 검출 평가 → train/val 목표 추상화 캐시 → screenshot 정책 5,000+5,000 steps →
abstract 정책 정답 추상화 5,000 steps + 예측 추상화 5,000 steps.
두 정책의 optimizer 초기화/학습률/step 예산은 동일하다. 검출기 추가 학습 비용은 별도다.
stage 2는 stage 1 best에서 시작하며 optimizer는 새로 만든다.
모든 detector/policy stage에 early stopping을 기본 적용한다. `EARLY_STOP_PATIENCE=10`은
validation loss가 10회 연속 개선되지 않으면 해당 단계를 정상 종료한다는 뜻이다.
기본 `EVAL_EVERY=500`에서는 보통 5,000 steps 동안 개선이 없는 경우다. `steps`는 최대 예산이다.
`EARLY_STOP_MIN_DELTA=0`이 기본이며, 양수이면 그 값보다 큰 절대 loss 감소가 있어야 patience를
초기화한다. `best.pt`는 min-delta와 무관하게 실제 최저 validation loss를 저장한다.
종료 시 `last.pt`에 patience와 종료 상태를 저장하고 다음 단계는 `best.pt`에서 시작한다.
같은 설정으로 재실행하면 early-stop 완료 stage는 건너뛴다. 새 stage의 patience는 초기화된다.
CLI 옵션은 `--early-stop-patience`, `--early-stop-min-delta`이며, patience=0이면 비활성화한다.
이 옵션들도 resume 설정 검사 대상이다. 기능 추가 전 run이나 다른 patience 설정을 사용하려면
새 output에서 `--init-checkpoint`로 가중치를 이관한다(optimizer와 patience는 새로 시작).
같은 명령을 재실행하면 데이터 캐시 및 각 단계 last checkpoint에서 재개한다.
데이터·설정이 바뀌면 새 output 디렉터리를 사용한다. v1 width/height/dx/dy 데이터는 v2 action
grammar와 호환되지 않으므로 반드시 새 data/run 디렉터리에서 다시 생성·학습한다. v1 policy를
`--init-checkpoint`로 지정하면 vision/tree/transformer 가중치만 이관하고 확장 action head는 새로 초기화한다.

기본 batch=2, accumulation=4, 정책 BF16, 검출기 FP32. OOM이면 `BATCH_SIZE=1`을 사용한다.
RTX 4090의 전체 시간/최대 메모리는 실측하지 않았다. elapsed_s 로그로 산출해야 한다.
최초 실행은 torchvision 사전학습 가중치를 다운로드한다. `--no-pretrained`는 smoke/ablation용이다.
학습 중 VLM은 필요하지 않다. 각 단계는 별도 프로세스로 실행한다.

```bash
# 실행 점검용: 이 모델로 정확도를 판단하지 않는다.
CUDA_VISIBLE_DEVICES=0 TRAIN_PAGES=30 DETECTOR_STEPS=2 \
POLICY_STAGE1_STEPS=2 POLICY_STAGE2_STEPS=2 BATCH_SIZE=1 ACCUMULATION=1 \
bash scripts/train_visual.sh data/visual-smoke runs/visual-smoke
```

설정: TRAIN_PAGES, TRAIN_HTML_MANIFEST, TRAJECTORIES, MAX_NOISE, BATCH_SIZE, ACCUMULATION,
DETECTOR_STEPS, POLICY_STAGE1_STEPS, POLICY_STAGE2_STEPS, RAW_SIZE, ABS_SIZE, PREDICTION_PROBABILITY, DEVICE.
기본 RAW_SIZE=ABS_SIZE=384로 해상도를 통제한다. 고해상도 raw 비교는 새 run에서
`RAW_SIZE=768 ABS_SIZE=384`를 사용하되 해상도 차이를 명시한다.
작은 요소 보존 ablation은 `ABS_SIZE=768`로 **데이터 생성부터** 다시 실행한다.

## 실제 웹 학습 데이터

### 사전학습부터 WebUI controlled test까지 한 번에 실행

```bash
CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=2 ACCUMULATION=4 \
WEBUI_TRAIN=6000 WEBUI_VAL=2000 WEBUI_TEST=2000 \
VAL_SAMPLES=2000 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_visual_full.sh data/raw/webui/test webui-10k-v2
```

이 스크립트는 procedural 1,000페이지 생성(800 train/100 val/100 test), detector 10,000 steps,
각 policy 10,000+5,000 steps를 실행한 뒤 WebUI detector 20,000 steps, 각 policy 30,000+15,000
steps로 fine-tuning한다. 이 step 수는 시작 예산이며 수렴이나 성능을 보장하는 값은 아니다.
사전학습은 `PRE_DETECTOR_STEPS`, `PRE_POLICY_STAGE1_STEPS`, `PRE_POLICY_STAGE2_STEPS`,
fine-tuning은 `FT_DETECTOR_STEPS`, `FT_POLICY_STAGE1_STEPS`, `FT_POLICY_STAGE2_STEPS`로 조정한다.

먼저 WebUI에 요청한 수만큼 중복 제거·host 분리 가능한 HTML이 있는지 확인한다. 전체 screenshot
파일 수만으로 6,000/2,000/2,000 분할 가능 여부를 보장할 수 없다. 분할 후 생성 실패 페이지는
제외하므로 `data/webui-10k-v2-webui/rendered/report.json`의 `split_coverage`에서 선택 수,
실제 사용 수, 제외 수를 확인한다. 선택한 test 2,000개 중 사용 가능한 **모든** 페이지를 평가하고,
누락을 감추거나 실패 페이지를 다른 split에서 보충하지 않는다. 실제 2,000개의 유효 평가 페이지가
필요하다면 먼저 데이터 생성 성공률을 확인하고 별도 분할 설계를 정해야 한다.

테스트는 저장된 손상 HTML과 재렌더링한 목표 화면을 쓰는 controlled 복구 실험이다.
원본 WebUI screenshot에서 VLM이 HTML을 생성하는 end-to-end 평가는 별도다.
결과는 `runs/webui-10k-v2-test/evaluation/report.md`, figure는
`runs/webui-10k-v2-webui/figures`에 저장한다. repeats=3이면 n은 실제 페이지 수의 3배다.

학습 스크립트의 validation 기본값은 고정된 512개 **중간 상태 예제**다. 페이지를 순회해 샘플링하므로
앞쪽 페이지에만 치우치지 않으며 `validation_pages`에 실제 페이지 수를 기록한다.
위 전체 실행 스크립트는 기본값을 2,000개로 늘려 최대 2,000개 validation 페이지를 포함한다.
`VAL_SAMPLES=1000000000`이면 모든 validation 예제를 평가하지만 주기적 검증 시간이 크게 늘어난다.
`EVAL_EVERY`로 검증 주기를 조절할 수 있다. 새 샘플링 규칙은 기존 run과 best loss를 혼합하지 않도록
resume 설정 검사에 포함되어 있으므로 이 버전은 새 run tag로 시작한다. 같은 tag·설정 재실행은 재개다.

Figure 생성에는 `pip install -e '.[visual,visual-metrics,test]'`가 필요하다.

기본 procedural 데이터는 bootstrap/smoke이며 실제 웹 일반화의 근거가 아니다.
렌더링 가능한 HTML을 준비해 JSONL manifest를 지정한다. 현재 브라우저는 HTML 문자열을
로드하므로 상대경로 assets는 먼저 data URI/절대 URL로 변환해야 한다. `rick.jpg` placeholder는
기존 helper로 삽입하지만 일반 asset 폴더 전체를 자동 번들링하지는 않는다:

```json
{"id":"train-001","group":"site-A","split":"train","html":"/datasets/train/site-A/page.html","viewport":[1280,720]}
{"id":"val-001","group":"site-B","split":"val","html":"/datasets/train/site-B/page.html","viewport":[1280,720]}
{"id":"test-001","group":"site-C","split":"test","html":"/datasets/train/site-C/page.html","viewport":[1280,720]}
```

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_HTML_MANIFEST=/datasets/train/manifest.jsonl \
bash scripts/train_visual.sh data/visual-real runs/visual-real
```

같은 site/template은 같은 group으로 묶는다. group/HTML 내용 hash의 split 중복을 거부하고
체크포인트에 학습 이력을 저장한다. 의미적으로 비슷한 template의 자동 검출은 아니다.
**Design2Code/Hard 시험 페이지를 학습에 넣지 않는다.** WebUI screenshot/AX만으로는
이 CSS corruption 학습 데이터를 만들 수 없다. HTML이 필요하다. WebUI 평가는 지원한다.

데이터 생성은 HTML을 변형해 정답과의 차이가 커지고 역편집이 직전 상태를 실제 브라우저에서
복원하는지 검증한다. 깨끗한 상태는 STOP이다. 두 정책은 동일 궤적/action label을 사용한다.
추상화 label은 선택된 가시 DOM 요소에서 생성하며 wrapper 중복을 줄이고 viewport에 clip한다.
완전한 DOM 역복원이 아니며 작은 글자·겹침·투명도·pseudo-element·node 수 제한이 남는다.
`detector-test.json`은 class-aware IoU@0.5 precision/recall/small recall이며 COCO AP는 아니다.

## 평가

### Held-out controlled HTML (먼저 실행)

```bash
CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 REPEATS=3 ORACLE_ABLATION=1 \
PAGE_MANIFEST=data/visual/rendered/pages-test.jsonl \
bash scripts/run_visual_web.sh unused runs/visual-controlled runs/visual
```

저장된 corrupted initial HTML을 사용하므로 VLM 다운로드/호출이 없다.
oracle ablation으로 목표 검출 오차와 정책 오차를 분리한다.

### Design2Code / Hard / WebUI

```bash
# 초기 HTML을 로컬 Qwen으로 생성할 때 한 번 설치 (학습에는 불필요)
pip install -e '.[vlm]'

CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=5 REPEATS=3 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/visual-d2c-smoke runs/visual

CUDA_VISIBLE_DEVICES=0 PAGE_LIMIT=0 REPEATS=3 \
bash scripts/run_visual_web.sh "$D2C_HARD_ROOT" runs/visual-hard runs/visual

CUDA_VISIBLE_DEVICES=0 WEB_DATASET=webui PAGE_LIMIT=5 REPEATS=3 \
bash scripts/run_visual_web.sh "$WEBUI_ROOT/test" runs/visual-webui-smoke runs/visual
```

VLM은 초기 HTML 생성에만 사용한다(기본 Qwen3-VL-8B-Instruct 4bit).
목표 box/계획은 만들지 않는다. VLM_BACKEND/VLM_MODEL/VLM_ENDPOINT,
MAX_PIXELS/MAX_NEW_TOKENS를 지정할 수 있다. 이미 생성한 HTML은 PAGE_MANIFEST에
id/screenshot/html(reference)/initial_html/viewport로 지정한다.
WebUI는 기록된 screenshot/AX를 reference로 사용한다. 공식 D2C metric과 같지 않다.
Design2Code에서만 기존 공식 evaluator clone을 OFFICIAL_REPO로 연결할 수 있다.

### Design2Code Self-Revision baseline

논문의 `Text-Augmented prompting -> Visual Self-Revision 1회`를 같은 initial 비교에 추가하려면
새 output 디렉터리에서 다음처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 SELF_REVISION_PROTOCOL=design2code PAGE_LIMIT=20 REPEATS=3 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/visual-d2c-self-revision runs/visual
```

이 옵션은 공식 코드와 같이 (1) reference HTML에서 추출한 줄 단위 텍스트와 목표 screenshot으로
initial HTML을 생성하고, (2) 목표 screenshot, initial 렌더링, initial HTML, 같은 텍스트를 입력해
전체 HTML을 한 번 수정한다. 결과 method는 `initial`과 `design2code-self-revision`이다.
기본 생성값도 논문의 `seed=2024`, `max_new_tokens=4096`, validation retry 0으로 바뀐다.
`SEED`, `MAX_NEW_TOKENS`, `VLM_RETRIES`로 명시적으로 덮어쓸 수 있다.

이 baseline은 **reference HTML의 oracle text를 사용**하므로 screenshot-only 결과와 입력 조건이
같지 않다. `uses_reference_text=true`가 prepared record와 evaluation metric row에 기록된다.
프롬프트·두 이미지 순서는 공식
[`gpt4v.py`](https://github.com/NoviScl/Design2Code/blob/main/Design2Code/prompting/gpt4v.py)를 따른다.
다만 기본 Qwen 실행은 방법의 재현이지 논문의 폐기된 `gpt-4-vision-preview` 모델 자체 재현은 아니다.
WebUI처럼 reference HTML이 없는 데이터에는 이 옵션을 사용할 수 없다.

## 결과와 해석

- `prepare/prepared.jsonl`: 공유 initial HTML/DOM, 생성 오류.
- `repair/results.jsonl`: initial/screenshot-policy/abstract-policy의 반복별 결과 HTML.
- `repair/timings.jsonl`, `timing-summary.json`: 목표 검출/인코딩, 현재 render/인코딩,
  policy/layout 시간, action 수, browser 실행/screenshot 수, 실패.
- `repair/pages/*/*trace.json`: 스텝별 action/box, 종료 사유.
- `evaluation/report.md`, `metrics.jsonl`: 최종 **실제 화면**의 pixel MAE/geometry IoU.

report의 n은 pages × repeats다. 반복을 독립 페이지처럼 통계 검정하면 안 된다.
timing-summary는 성공 trial 평균과 실패 건수를 분리한다. report는 실패 fallback도 포함한다.
repair 시간에는 목표 검출/인코딩을 포함하고 model load/warmup/최종 평가 캡처는 제외한다.
report pipeline에는 공유 초기 HTML 생성/DOM 준비도 포함한다.
반복 중 screenshot이 0이어도 초기 준비/최종 평가 screenshot까지 0인 것은 아니다.

기본은 동일 최대 action 예산 REPAIR_STEPS=20이며 STOP 때문에 실제 step은 다를 수 있다.
새 output에서 TIME_BUDGET=2처럼 지정하면 soft 시간 예산을 비교한다. 연산 사이 검사이므로
한 번의 검출/인코딩/브라우저 실행만큼 초과할 수 있고 초과량을 기록한다.
한 번의 검출 비용 때문에 짧은 rollout에서는 추상화 방식이 더 느릴 수도 있다.
accuracy와 시간을 함께 보고해야 한다.

## 논문용 학습 figure

새로 시작한 v2 run은 각 stage의 `train.jsonl`에 train/validation loss와 함께 다음 검증 지표를
영구 저장한다.

- detector: loss component, class-aware precision/recall@IoU 0.5, 32×32 미만 small-element recall
- policy: full action, node, CSS property, value 및 STOP 정확도
- policy: width/height/margin/padding/gap/flex/justify/align별 정확도와 표본 수
- controlled rollout: step별 geometry IoU와 relation accuracy. 이 정답 지표는 action 선택에는
  사용하지 않고 figure를 위한 사후 진단에만 사용한다.

학습과 controlled 평가가 끝나면 PNG(300 dpi)와 벡터 PDF를 함께 생성한다.

```bash
python -m framediff visual-figures \
  --run-root runs/visual-webui-v2 \
  --evaluation runs/eval-webui-controlled-v2/repair \
  --out runs/visual-webui-v2/figures
```

생성물은 `training-curves`, `detector-validation`, `action-accuracy`,
`per-property-accuracy`, `denoising-trajectory`의 `.png`/`.pdf`와 수치가 담긴 JSON이다.
`--evaluation`은 선택 사항이며 생략하면 denoising trajectory만 만들지 않는다. 이전 버전 로그에는
세부 검증 지표가 없으므로 새 figure의 action/detector panel을 얻으려면 현재 코드로 다시 학습해야
한다. 서로 다른 stage의 loss scale은 같다고 가정하지 말고 각 panel 안에서 수렴과 best validation
step을 해석한다. 논문 표의 최종 성능은 별도의 held-out evaluation 결과를 사용한다.

현재 범위는 동일 DOM에서 위치·크기·정렬·공간 관계 회복이다. DOM reparenting/새 요소 생성,
responsive CSS 전체 복원, 실제 benchmark 성능 향상은 보장하지 않는다.
