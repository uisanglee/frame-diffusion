# TUIDE 정책 모델 크기 비교

모든 크기의 모델은 고정된 detector, 의미 채널, CSS 문법, 이미지 크기,
CSS owner 최대 512개, 고정된 val/test 손상 데이터, 준비된 목표 캐시를 공유합니다.
달라지는 것은 hidden 차원, 트리/디코더 깊이, attention head 수뿐입니다.

| 설정 | Hidden | Layers | Heads | Abstract 정책 파라미터 수 |
|---|---:|---:|---:|---:|
| s | 128 | 3 | 4 | 4,766,449 |
| m | 256 | 6 | 8 | 15,532,325 |
| l | 512 | 8 | 8 | 67,087,477 |

이 수치에는 정책 모델의 공간 이미지 인코더가 포함됩니다. 고정된 detector의
파라미터 41,314,536개는 별도입니다. 현재 스타일시트 문법과 owner 512개를
가정한 수치입니다. 모델을 크게 해도 지원되는 CSS 액션이 늘어나지는 않습니다.

## 물리 GPU 1, 3, 4에서 실행

저장소 루트에서 Python 환경을 활성화한 뒤 실행합니다. 다음 명령어는 기존 v3
데이터와 detector 캐시를 사용하므로 별도의 전처리 명령어가 필요하지 않습니다.
다만 학습 중 새로운 online corruption 상태는 계속 렌더링합니다.

온라인 CPU worker는 페이지를 독립적으로 준비합니다. 샘플 하나가 타임아웃되면
해당 worker와 그 브라우저 자식만 재시작하며, 다른 worker의 대기·완성 결과는
유지합니다. 학습에는 기존 샘플 인덱스 순서대로 전달합니다. `ONLINE_TIMEOUT`
(`run_webui_scale.sh` 기본 900초)은 worker 시작을 포함한 실제 작업 시간에
적용됩니다. 다른 프로세스가 생성하는 공유 캐시의 잠금 대기는 별도의
`TUIDE_CACHE_LOCK_TIMEOUT`(기본 3600초)으로 제한합니다. 타임아웃 로그의
`stage`로 `working`과 `cache_wait`를 구분할 수 있습니다. CPU·메모리 여유가
있을 때 `ONLINE_WORKERS`를 늘리고, `ONLINE_PREFETCH`로 대기 결과 수를 제한하세요.

기존 성공 공유 캐시는 그대로 재사용합니다. 수정 가능한 CSS가 없거나 고정된
후보 검사에서 시각적 효과를 찾지 못한 확정 실패는 같은 소스·설정 키에
캐시하여 반복 준비를 막습니다. 일시적인 브라우저·검사 오류는 다시 시도합니다.
소스나 전처리 기준이 바뀌면 실패 캐시도 무효화됩니다. full-DOM 측정은 브라우저
질의 한 번으로 묶었으며 박스·후보 판정·teacher 기준은 같습니다. Train 전체
사전 생성이나 val/test 재생성은 필요하지 않습니다.

```bash
mkdir -p log
CUDA_VISIBLE_DEVICES=1 nohup bash scripts/train_tuide_scale.sh s > log/tuide-s.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup bash scripts/train_tuide_scale.sh m > log/tuide-m.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup bash scripts/train_tuide_scale.sh l > log/tuide-l.log 2>&1 &
```

기본값은 배치 크기 2, gradient accumulation 4, seed 42입니다. Stage 1은
학습률 1e-4로 최대 30,000 step, stage 2는 학습률 1e-5로 최대 15,000 step
진행합니다. 500 step마다 validation 샘플 2,000개를 평가하며, early-stop
patience는 10입니다. 각 stage에 `last.pt`가 있으면 이를 불러옵니다. Stage
2에 `last.pt`가 없으면 해당 크기 모델의 stage 1 `best.pt`에서 시작합니다.
모델 크기나 실행 설정을 바꿀 때는 출력 디렉터리를 새로 사용하세요. 여기서의
학습 조기 종료는 validation loss 기준이며, 아래의 **추론 중 픽셀 조기 종료와
별개**입니다.

Validation의 최적 체크포인트 선택과 학습 조기 종료에는 기존 token CE를 계속
사용합니다. 기존 고정 128개 진단 샘플에서는 `action_target_accuracy`(예측한
owner·속성이 실제 손상 위치 중 하나인 비율)와 `exact_repair_action_rate`(한
액션으로 CSS 오류 개수를 줄인 비율)를 함께 기록합니다. 같은 예측을 재사용하므로
추가 렌더링이나 detector 실행은 없습니다. `480px`와 `480.0px`처럼 같은 단위의
숫자 표기는 동등하게 처리하며, 단위 없는 길이 0도 `0px`와 같습니다. 단위·퍼센트·
키워드·`important` 차이는 유지합니다. 정답 teacher 하나뿐 아니라 현재 손상된
모든 owner·속성을 올바른 수정 대상으로 인정합니다.

`action_metric_contract=css-action-diagnostics-numeric-v1`으로 새 비교 기준을
표시합니다. 기존 `symbolic_*`도 이 정규화를 적용하므로 문자열 그대로 비교하던
이전 로그와 구분해야 합니다. `symbolic_error_count_unchanged_rate`는 기존
`symbolic_no_change_action_rate`의 의미를 명확하게 나타낸 이름입니다. 오류 개수가
그대로여도 CSS 값이나 화면은 부분 개선될 수 있습니다. 속성별 `n`은 모델의 예측
횟수이며 손상 데이터 개수가 아닙니다. `target_accuracy`와 `exact_repair_rate`도
각 속성의 예측 횟수를 분모로 씁니다. 실행할 수 없는 값은 정확 복구로 인정하지
않지만, 유효한 손상 위치를 골랐다면 대상 선택은 맞을 수 있습니다. 기존 캐시와
체크포인트를 사용해 재개할 수 있고 추가 전처리는 필요하지 않습니다.

필요하면 경로를 명시적으로 지정할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=3 bash scripts/train_tuide_scale.sh m \
  data/webui-css-owners-v3 \
  runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
  runs/css-owners-v3-shared \
  runs/tuide-m-abstract
```

최종 체크포인트 경로는 다음과 같습니다.

```text
runs/tuide-s-abstract/replacement/abstract-stage2/best.pt
runs/tuide-m-abstract/replacement/abstract-stage2/best.pt
runs/tuide-l-abstract/replacement/abstract-stage2/best.pt
```

기본 유효 배치 크기는 8입니다. L 모델이 GPU 메모리를 초과하면 새 실행에서
`BATCH_SIZE=1 ACCUMULATION=8`을 사용하세요. 크기별로 유효 배치와 학습
예제 예산을 비슷하게 유지하고 실제 step 수와 시간을 보고해야 합니다. 최대
step 수가 같더라도 조기 종료가 발생하면 학습 연산량이 같지는 않습니다. 주
모델 크기는 test가 아니라 validation 성능으로 선택합니다.

## RGB 비교 및 평가

통합 크기 비교 스크립트에는 S screenshot 체크포인트와 S/M/L abstract
체크포인트가 필요합니다. 초기 HTML을 한 번 준비한 뒤, S는 동일한 크기의 RGB
baseline과 비교하고 M/L은 별도 프로세스에서 abstract-only로 평가합니다.
M/L screenshot 학습은 선택 사항입니다.

```bash
CUDA_VISIBLE_DEVICES=1 TUIDE_SCALES=1 PAGE_LIMIT=20 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/eval-tuide-scales \
  runs/webui-10k-v3-nospacing-fresh-webui
```

이미 평가 페이지를 준비했다면 VLM 준비 단계를 생략할 수 있습니다.

```bash
CUDA_VISIBLE_DEVICES=1 PAGE_LIMIT=0 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_tuide_scales.sh \
  runs/existing-eval/prepare/prepared.jsonl runs/eval-tuide-scales
```

통합 보고서는 크기 비교 출력 폴더 아래의 `comparison/paper-report.md`에
생성됩니다. `run_visual_web.sh`를 거치면 `scales/` 아래에 생성됩니다. 가능한
경우 initial/self-revision 결과, screenshot-policy(S), TUIDE-S/M/L 결과가
포함됩니다. `policy-parameters.json`에는 측정된 파라미터 수가 기록됩니다.
반복 결과는 페이지별로 평균을 냅니다. 필요한 경우 `RAW_CHECKPOINT`,
`DETECTOR_CHECKPOINT`, `TUIDE_S_CHECKPOINT`, `TUIDE_M_CHECKPOINT`,
`TUIDE_L_CHECKPOINT`로 경로를 지정합니다. 크기 비교 실행은 동일한 prepared
JSONL, seed, step 예산, detector를 공유합니다. S RGB와 M/L abstract를
비교하면 입력 방식과 모델 크기가 동시에 바뀌므로, abstraction만의 효과를
분리하려면 S RGB와 S abstract를 비교하세요.

### 추론 중 픽셀 목표에 따른 조기 종료

screenshot과 TUIDE abstract 복구는 기본적으로 현재 브라우저 렌더와 목표
스크린샷을 **원본 해상도**에서 비교합니다. 전체 픽셀의 99% 이상에서 RGB
각 채널의 정규화된 차이가 모두 0.005 이하면 조기 종료합니다. 첫 액션 전과
이후 액션 후에 검사하며, `REPAIR_STEPS`가 허용한 마지막 액션 직후에도
검사합니다. 따라서 `REPAIR_STEPS`의 기본값 20은 **최대 step 수**이지
반드시 실행하는 수정 횟수가 아닙니다. 이미지 오차가 악화되었다는 이유로
액션을 되돌리지는 않습니다. 스크롤 보존을 위한 별도의 guard는 유지됩니다.
평가 결과에는 `goal_reached`, `stop_reason`, `pixel_match_fraction`, 픽셀
목표 검사 횟수와 시간이 기록됩니다. 이는 공개 TinySVG 코드의 Gaussian 점수
판정과 달리, 허용 오차를 만족한 픽셀의 비율을 세는 방식입니다.

일반 평가 경로(`TUIDE_SCALES=1`을 지정하지 않은 경우)의
`run_visual_web.sh`에서는 `PIXEL_GOAL_FRACTION`의 기본값이 `0.99`,
`PIXEL_GOAL_TOLERANCE`의 기본값이 `0.005`입니다. 예를 들어 99.5% 일치를
요구하려면 실행 명령어의 환경변수에 `PIXEL_GOAL_FRACTION=0.995`를 추가합니다.
고정 step 비교가 필요하면 `PIXEL_GOAL_FRACTION=-1`로 픽셀 조기 종료를
끕니다. `visual-evaluate`와 `visual-test-rollout`을 직접 실행할 때는 각각
`--pixel-goal-fraction`, `--pixel-goal-tolerance` 옵션을 사용합니다.
`run_tuide_scales.sh` 경로(`TUIDE_SCALES=1`일 때도 사용)는 CLI 기본값을
사용하며 이 환경변수 override를 전달하지 않습니다. 비교 실험에서는 픽셀
목표 설정을 동일하게 맞추세요.
설정을 바꾼 뒤 동일한 평가 출력 디렉터리에서 `--resume`하는 것은 지원하지
않습니다.

동일한 크기의 RGB 정책 모델은 다음처럼 학습합니다.

```bash
CUDA_VISIBLE_DEVICES=1 POLICY_MODES=screenshot bash scripts/train_tuide_scale.sh m
```

`visual-evaluate`는 체크포인트에서 모델 차원을 읽습니다. RGB/abstract를
함께 평가할 때는 차원이 같아야 합니다. `--abstract-only`는 RGB 체크포인트
없이 단일 크기 모델을 평가합니다. 선택적인 M 크기 비교 예시는 다음과
같습니다.

```bash
CUDA_VISIBLE_DEVICES=1 \
RAW_CHECKPOINT=runs/tuide-m-screenshot/replacement/raw-stage2/best.pt \
ABSTRACT_CHECKPOINT=runs/tuide-m-abstract/replacement/abstract-stage2/best.pt \
DETECTOR_CHECKPOINT=runs/webui-10k-v3-nospacing-fresh-webui/detector/best.pt \
PAGE_LIMIT=20 REPEATS=3 REPAIR_STEPS=20 \
bash scripts/run_visual_web.sh "$D2C_ROOT" runs/eval-tuide-m \
  runs/webui-10k-v3-nospacing-fresh-webui
```

모델 크기별 성능을 통제해 비교하려면 `visual-evaluate --data ...`로 완전히
동일한 prepared 평가 레코드와 초기 HTML을 재사용하세요. 크기별로 초기 HTML을
각각 다시 생성하지 마세요. 최종 rollout geometry, 복구 시간, 실패 건수,
학습 연산량, 정책 모델 파라미터 수를 보고합니다.

## 직접 실행하는 CLI

### 입력 표현 ablation

`train_tuide_scale.sh`의 `ABSTRACTION_REPRESENTATION` 또는
`visual-tree-train`의 `--abstraction-representation`으로 입력을 선택합니다.

| 값 | 한 번의 관측값 | Target/current/절댓값 차이 입력 |
|---|---|---|
| `abstract-rgb` | 고정된 클래스 색상으로 채운 박스, ImageNet 정규화 RGB | 9채널 |
| `occupancy` | 클래스별 채움 마스크 4개 | 12채널 |
| `boundary` | 클래스별 박스 경계 마스크 4개 | 12채널 |
| `occupancy-boundary` (기본값) | 채움 마스크와 경계 마스크 모두 사용 | 24채널 |

`boundary`는 원본 스크린샷의 edge detector가 아닙니다. 감지된 박스 또는
DOM 박스 각각의 경계를 보존하므로, 같은 클래스 박스가 겹쳐도 내부 경계를
나타냅니다. `abstract-rgb`는 동일한 박스를 정해진 그리기 순서로 색칠하며,
웹페이지의 원본 RGB를 사용하지 않습니다. 이 방식은 겹치는 요소를 가릴 수
있지만, 마스크 방식은 클래스별 점유 영역과 경계를 보존합니다. 네 설정 모두
입력 convolution을 제외하면 같은 ResNet-18과 복구 디코더를 사용합니다.
따라서 이는 완전히 다른 인코더 구조가 아니라 **입력 표현의 차이**를
실험합니다.

동일한 준비된 정책 데이터, detector 체크포인트, 고정된 예측 목표 캐시를
재사용합니다. 이 ablation을 위해 라벨을 다시 만들거나 이미지를 다시 생성할
필요는 없습니다. 입력 표현마다 다른 출력 디렉터리를 사용하세요. 예를 들어
`POLICY_DATA`를 기존 `.../7000/css`, `DETECTOR`를 detector의 `best.pt`,
`SHARED_RUN`을 `predicted-train/data.jsonl`과 `predicted-val/data.jsonl`이
있는 기존 디렉터리로 설정한 뒤 다음 명령어로 네 설정을 순차 실행합니다.

```bash
for rep in abstract-rgb occupancy boundary occupancy-boundary; do
  env CUDA_VISIBLE_DEVICES=2 POLICY_MODES=abstract \
    ABSTRACTION_REPRESENTATION="$rep" BATCH_SIZE=4 ACCUMULATION=2 \
    SEED=42 POLICY_STAGE1_STEPS=30000 POLICY_STAGE2_STEPS=15000 \
    EARLY_STOP_PATIENCE=0 \
    bash scripts/train_tuide_scale.sh s "$POLICY_DATA" "$DETECTOR" "$SHARED_RUN" \
      "runs/tuide-input-ablation/7000/$rep" || break
done
```

Stage 2는 같은 입력 표현의 stage 1에서 초기화됩니다. 기존 `last.pt`로는
설정이 일치할 때만 재개할 수 있습니다. 다른 표현으로 학습된 정책 모델에서
새 표현의 모델을 초기화하지 마세요. 기존 기본 체크포인트는
`occupancy-boundary`로 읽을 수 있습니다. 평가 시에는 체크포인트에서 입력
표현을 자동으로 읽으므로 별도 override가 필요하지 않습니다. 입력 표현별로
동일한 고정 test 상태, detector threshold, 복구 step 예산, seed를
사용하세요. 추론 종료 판정에는 모든 표현에서 동일한 원본 해상도의
목표-현재 RGB 픽셀 기준을 사용하며 teacher action 생성 방식은 변하지
않습니다. optimizer step 예산이 같아도 FLOPs가 같다는 뜻은 아니므로
실행 시간과 파라미터 수도 보고하세요. 최종 비교에는 여러 seed를 사용하세요.

`visual-tree-train --policy-scale s|m|l`은 실행 및 체크포인트 검증 전에
설정값을 결정하며, 명시적으로 지정한 `--hidden`, `--layers`, `--heads` 값보다
우선합니다. preset을 지정하지 않으면 기존 기본값이나 사용자 지정 차원을
사용할 수 있습니다. 다른 설정과 입력 서명이 일치한다면 실제 결정된 차원만
재개 실행의 동일성 판정에 사용하므로 기존 S 실행과의 호환성이 유지됩니다.
