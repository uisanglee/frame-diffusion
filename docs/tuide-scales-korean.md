# TUIDE S/M/L 학습과 평가

현재 경로는 CSS 우선 적용 WebUI 전처리와 혼합 손상 학습입니다.
이전 screened 7k/40k·inline-only 실행 안내는 제거했습니다.
[전처리](webui-css-first-preparation.md), [혼합 손상](mixed-corruption.md),
[영문 상세 설정](tuide-scales.md)을 함께 참고하세요.

| 모델 | Hidden | Layers | Heads | Abstract 정책 파라미터 수 |
|---|---:|---:|---:|---:|
| S | 128 | 3 | 4 | 4,766,449 |
| M | 256 | 6 | 8 | 15,532,325 |
| L | 512 | 8 | 8 | 67,087,477 |

스타일시트 문법과 CSS owner 최대 512개 기준이며 정책 이미지 인코더를 포함합니다.
고정 Detector의 41,314,536개 파라미터는 별도입니다. 모든 크기는 동일한 편집
문법·데이터·Detector·고정 Validation/Test를 사용합니다.

## 공통 경로

```bash
cd /home/uisang/2027/frame-diffusion
source .venv/bin/activate
mkdir -p log
export TRAIN_DATA="$PWD/data/new-css-v1/7000/css"
export TRAIN_RUN="$PWD/runs/webui-css-mixed-v1/7000"
export DETECTOR_CHECKPOINT="$TRAIN_RUN/detector/best.pt"
```

학습 스크립트의 기본 경로도 위와 같습니다. 명령 뒤에 전달하는 데이터·Detector·
공유 캐시·출력 인자가 우선합니다. 기존 실행을 재개할 때는 원래 사용한 경로를
명시하세요. CSS 적용으로 Target이 바뀌거나 mixed/STOP 학습으로 전환할 때는
기존 체크포인트를 이어 학습하지 않고 새 run에서 시작합니다.

## Detector 및 Stage1

```bash
CUDA_VISIBLE_DEVICES=2 STAGE=detector \
  bash scripts/run_webui_scale.sh 7000 data/new-css-v1 runs/webui-css-mixed-v1

CUDA_VISIBLE_DEVICES=0 POLICY_STAGES=1 bash scripts/train_tuide_scale.sh s
```

Stage1은 정답 DOM 추상화를 사용하므로 Detector 없이 시작할 수 있습니다.
`s`를 `m`, `l`로 바꾸면 각 크기를 학습합니다.
스크린샷 정책은 `POLICY_MODES=screenshot POLICY_STAGES=1`이며 Stage2가 없습니다.

## Detector 캐시와 Stage2

같은 데이터 버전의 Detector 학습 후 아래 캐시를 한 번 준비합니다.
이미 완료한 항목은 재사용합니다.

```bash
for split in train val; do
  CUDA_VISIBLE_DEVICES=2 python -m framediff visual-cache-targets \
    --data "$TRAIN_DATA/policy-$split.jsonl" \
    --checkpoint "$DETECTOR_CHECKPOINT" \
    --out "$TRAIN_RUN/shared/predicted-$split" \
    --device cuda --size 384 --resume
done
```

각 크기의 Stage1과 위 캐시가 완료되면:

```bash
nohup setsid env CUDA_VISIBLE_DEVICES=0 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh s > log/tuide-s-stage2.log 2>&1 < /dev/null &
nohup setsid env CUDA_VISIBLE_DEVICES=1 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh m > log/tuide-m-stage2.log 2>&1 < /dev/null &
nohup setsid env CUDA_VISIBLE_DEVICES=2 POLICY_STAGES=2 \
  bash scripts/train_tuide_scale.sh l > log/tuide-l-stage2.log 2>&1 < /dev/null &
```

Stage2는 같은 크기·표현의 Stage1 `best.pt`에서 시작합니다. 해당 Stage2의
`last.pt`가 있으면 이를 재개합니다. 결과는
`$TRAIN_RUN/tuide-{s,m,l}-abstract/replacement/abstract-stage2/best.pt`입니다.

기본 batch 2, accumulation 4, seed 42이며 Stage1 최대 30,000 step,
Stage2 최대 15,000 step입니다. 500 step마다 최대 2,000 validation 샘플을
평가하고 token CE로 최적 체크포인트·조기 종료를 결정합니다. 손상 위치 선택·
정확 복구·손상 강도별 진단은 별도로 기록합니다. 실제 rollout 개선을 CE만으로
판정하지 마세요.

## 캐시와 프로세스

HTML·viewport·검사 설정이 같으면 S/M/L과 screenshot/abstract가 완료된 검사
캐시를 공유합니다. 새 손상 상태는 학습 중 생성하며 고정 Validation/Test는
재사용합니다. 소스·설정이 바뀐 라벨을 그대로 가져오지는 않습니다.

`ONLINE_WORKERS`와 `ONLINE_PREFETCH`는 CPU/RAM에 맞춰 조정합니다.
`ONLINE_TIMEOUT`은 실제 작업 시간이며 공유 캐시 잠금 대기는 별도 제한을
사용합니다. 학습 중단 시 해당 Python PID에 SIGTERM을 보내 worker 정리가
끝나도록 기다리세요. 재개는 마지막 저장된 체크포인트부터이며 미저장 step은
복구되지 않습니다.

## 평가와 ablation

```bash
CUDA_VISIBLE_DEVICES=0 REPAIR_STEPS=20 REPEATS=1 \
  bash scripts/run_tuide_scales.sh \
    runs/existing-eval/prepare/prepared.jsonl runs/eval-tuide-scales
```

모든 크기는 저장된 동일 Initial HTML을 사용합니다. S screenshot과 S abstract를
비교해야 표현 방식만 분리할 수 있습니다. M/L까지 비교하면 모델 용량도 바뀝니다.
체크포인트별 경로 변경 방법과 표현 ablation은 [상세 설정](tuide-scales.md)에 있습니다.

위치·크기 기반 보호는 `--preserve-matched-geometry`로 명시적으로 켭니다.
기존 scale runner가 이를 자동으로 켜거나 기존 결과를 바꾸지는 않습니다.
[기하 보호 진단](geometry-preservation.md)에서 별도 출력으로 비교하세요.
이 보호 기능과 mixed 학습의 STOP 예측, validation 기반 학습 조기 종료는
서로 다른 기능입니다.
