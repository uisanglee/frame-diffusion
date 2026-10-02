# WebUI visual fine-tuning

이 경로는 LLM/VLM 없이 WebUI에서 detector, screenshot policy, abstract policy를 모두
fine-tuning한다. 기본 주 실험은 procedural checkpoint에서 시작하고, 동일 데이터의 WebUI-only
초기화 실험을 ablation으로 둔다.

## 데이터 계약

`visual-import-webui`는 한 viewport의 screenshot/HTML/AX/box를 찾고 HTML hash를 중복 제거한 뒤
URL hostname 단위로 600/200/200을 분할한다. 한 domain은 둘 이상의 split에 들어가지 않는다.

```bash
python -m framediff visual-import-webui \
  --root "$WEBUI_ROOT/train" --out data/visual-webui/source \
  --view default_1280-720 --train-count 600 --val-count 200 --test-count 200
```

가능하면 공식 WebUI train split을 사용한다. 다운로드한 `test_split`을 600/200/200으로 다시
나눠 학습할 수도 있지만, 그 결과는 공식 WebUI test 성능이 아니라 custom WebUI-derived split이다.

생성 파일:

- `manifest.jsonl`, `manifest-{train,val,test}.jsonl`: HTML corruption/policy 데이터용.
- `native-detector-*.jsonl`: 원본 screenshot + AX text/image/control box.
- `report.json`: 페이지/domain/class 수와 제외 원인.

AX에는 CSS painted-region 정답이 없으므로 native detector 파일은 불완전한 3-class 보조 ablation이다.
기본 학습은 HTML을 Chromium으로 재렌더링해 네 class를 동일한 규칙으로 붙인 `rendered` detector
데이터를 사용한다.

## Procedural → WebUI fine-tuning

기존 procedural run을 보존한 채 새 디렉터리에서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 \
WEBUI_TRAIN=600 WEBUI_VAL=200 WEBUI_TEST=200 \
DETECTOR_STEPS=5000 POLICY_STAGE1_STEPS=3000 POLICY_STAGE2_STEPS=3000 \
bash scripts/finetune_visual_webui.sh \
  "$WEBUI_ROOT/train" \
  data/visual-webui-ft \
  runs/visual-webui-ft \
  runs/visual
```

초기 checkpoint:

- detector ← `runs/visual/detector/best.pt`
- screenshot policy ← `runs/visual/policy-raw/best.pt`
- abstract policy ← `runs/visual/policy-abstract/best.pt`

fine-tuning은 optimizer를 새로 만들고 기존 checkpoint의 학습 group/hash provenance를 보존한다.
기본 LR은 detector/policy stage 1이 `3e-5`, stage 2가 `1e-5`다. 기존 run과 architecture,
`RAW_SIZE`, `ABS_SIZE`가 같아야 한다. 각 stage의 `last.pt`가 있으면 정확히 resume한다.
structured-action v2로 전환할 때는 기존 v1 policy의 공통 encoder/transformer만 초기화에 재사용되고
확장된 action head는 새로 학습된다. 기존 v1 rendered data/run 디렉터리를 resume하지 말고 새 경로를 쓴다.
detector는 action grammar와 독립적이므로 이미 fine-tune한 run을 네 번째 인자로 주고
`REUSE_DETECTOR=1`을 설정하면 detector 학습을 건너뛸 수 있다:

```bash
CUDA_VISIBLE_DEVICES=1 REUSE_DETECTOR=1 \
POLICY_STAGE1_STEPS=3000 POLICY_STAGE2_STEPS=3000 \
bash scripts/finetune_visual_webui.sh \
  "$WEBUI_ROOT/test" data/visual-webui-v2 runs/visual-webui-v2 runs/visual-webui-ft
```

HTML 재렌더링과 원본 WebUI screenshot의 `source_pixel_mae`가 각 held-out page에 기록된다.
외부 CSS/asset 누락 페이지를 필터링하려면 새 data/run 디렉터리에서 예를 들어
`MAX_SOURCE_MAE=0.25`를 지정한다. 먼저 기본값 1.0으로 작은 subset을 만들어 MAE 분포를 확인하는
것이 안전하다. `MIN_ELEMENTS` 기본값은 3이다.

## WebUI-only ablation

같은 split에서 procedural checkpoint 없이 ImageNet backbone + 새 head/policy로 시작한다.

```bash
CUDA_VISIBLE_DEVICES=0 FINETUNE_FROM_PROCEDURAL=0 \
WEBUI_TRAIN=600 WEBUI_VAL=200 WEBUI_TEST=200 \
bash scripts/finetune_visual_webui.sh \
  "$WEBUI_ROOT/train" \
  data/visual-webui-only \
  runs/visual-webui-only \
  unused
```

공정 비교를 위해 importer seed, split counts, steps, batch/accumulation을 주 실험과 동일하게 둔다.
두 data directory의 `source/manifest*.jsonl` SHA가 같은지도 확인한다.

## Native AX detector ablation

원본 WebUI screenshot의 AX text/image/control box로 detector를 fine-tuning하려면
`DETECTOR_SOURCE=native`를 지정한다. painted-region이 unlabeled background로 취급될 수 있으므로
주 결과가 아니라 ablation으로 보고한다.

```bash
DETECTOR_SOURCE=native bash scripts/finetune_visual_webui.sh \
  "$WEBUI_ROOT/train" data/visual-webui-native runs/visual-webui-native runs/visual
```

## 평가

동일 checkpoint를 WebUI-derived controlled test, Design2Code 484, Design2Code-Hard 80에 평가한다.
D2C/Hard에서만 초기 HTML 생성에 Qwen이 사용되며 학습에는 사용되지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 REPEATS=3 \
bash scripts/evaluate_webui_finetune.sh \
  data/visual-webui-ft \
  runs/visual-webui-ft \
  "$D2C_ROOT" \
  "$D2C_HARD_ROOT" \
  runs/eval-visual-webui-ft
```

smoke test는 `CONTROLLED_LIMIT=20 D2C_LIMIT=20 HARD_LIMIT=20 REPEATS=1`을 추가한다.
각 외부 평가의 성공 initial HTML 수와 failure rate를 함께 보고하고, D2C 공식 metric은
`OFFICIAL_REPO`가 지정됐을 때만 계산된다.

## 결과 해석

필수 비교는 다음 세 개다.

1. procedural only: 기존 `runs/visual`.
2. WebUI only: `FINETUNE_FROM_PROCEDURAL=0`.
3. procedural → WebUI: 기본 fine-tuning.

WebUI corruption test는 동일 DOM의 geometry 복구 능력이고, D2C/Hard는 Qwen initial HTML의 구조·내용
오류까지 섞인 외부 일반화 평가다. 두 결과를 같은 종류의 정확도로 해석하면 안 된다.
