# 로컬 검증 기록

실행일: 2026-09-30, macOS arm64, Python 3.12, PyTorch 2.14.0 CPU.

## 검증된 항목

- `pytest`: **13 passed**. 단위·통합·Chromium 테스트 포함.
- 150개 합성 페이지 생성: train 121 / val 15 / test 14.
- hidden 64, 2 layers, 4 heads, 702,584 parameters, batch 4 모델 학습.
- 50-step 학습 후 `last.pt`에서 2,000-step까지 재개.
- checkpoint 로드와 inference, 14개 test 페이지 × 3 viewport × 5 방법 평가.
- 3개 test 페이지에서 proxy / browser-in-loop 비교.
- 1-shot box 모델의 별도 학습·평가 통합 테스트.
- WebUI 실제 파일 schema를 모사한 fixture의 import/fit 테스트.
- predicted-frame 입력과 독립 평가 정답이 분리되는지 테스트.
- VLM API adapter의 prompt/JSON 입출력은 **mock**으로 테스트.
- 패키지 editable install 및 CLI 실행.
- smoke suite 학습→평가→seed 집계 실행.

## 소규모 결과 (성능 논문의 결과가 아님)

14개 테스트 페이지, candidate budget 160, beam 2, top-k 8, 최대 10 step:

| 방법 | 평균 box IoU | 수정 단계 CPU 초/페이지 |
|---|---:|---:|
| 무수정 | 0.9392 | <0.001 |
| coordinate search | 0.9778 | 0.147 |
| random proposal + selection | 0.9648 | 0.129 |
| 학습 모델 + proxy | 0.9993 | 0.048 |
| 목표 box 복사 (코드 없음) | 1.0000 | <0.001 |

모든 실행 코드 결과에서 경량 executor와 Chromium box 간 최대 좌표 차이는 **0.0302734375 px**였습니다. 실행별 상세 결과는 `runs/smoke/eval/`에 있습니다.

별도 3개 샘플, 최대 5 step, budget 80:

| 피드백 실행기 | 수정 단계 초/페이지 | 브라우저 실행/페이지 (최종 검증 포함) |
|---|---:|---:|
| 경량 proxy | 0.034 | 3 |
| 매 후보 Chromium | 0.385 | 86 |

`runs/smoke/latency/`에 원본 결과·스크린샷이 있습니다. 두 방법 모두 해당 3개 샘플 IoU 1.0. 브라우저 방법은 screenshot/VLM 호출 없이 geometry만 실행합니다. 실제 raster+VLM 비용이나 GPU에서의 가속률로 확대 해석할 수 없습니다. warm-up·백그라운드 부하를 엄격히 통제한 latency benchmark도 아닙니다.

## 재현 명령

```bash
source .venv/bin/activate
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/ms-playwright"
python -m framediff generate --out data/smoke --count 150 --seed 42
python -m framediff train --train data/smoke/train.jsonl --val data/smoke/val.jsonl \
  --out runs/reproduce --steps 2000 --batch-size 4 --hidden 64 --layers 2 \
  --heads 4 --eval-every 250 --log-every 250 --val-samples 12 --device cpu
python -m framediff evaluate --data data/smoke/test.jsonl \
  --checkpoint runs/reproduce/best.pt --out runs/reproduce/eval \
  --methods none,coordinate,random,model,copy-boxes --steps 10 --beam 2 \
  --topk 8 --budget 160 --device cpu --browser-verify
```

위 새 학습은 한 번에 2,000-step을 수행합니다. 실제 저장된 run은 50-step 후 재개했고 validation cadence가 달라 dropout RNG 소비 순서도 달라질 수 있으므로, bitwise 동일한 checkpoint를 주장하지 않습니다. epoch, 소요 시간, checkpoint 선택이 달라질 수 있습니다.

## 아직 검증하지 않은 항목

- RTX 4090에서 BF16/4-bit 실행, VRAM, 처리량, 장시간 학습.
- 실제 Qwen weight 다운로드 및 추론. 외부 API 서버 실제 호출.
- 대규모 WebUI 아카이브 import와 fitting의 수율·실제 일반화.
- 실제 AR 오류 train/test 분리와 end-to-end 성능.
- 템플릿/사이트 holdout, 색상·텍스트·이미지·기능의 보존.
- 원 논문/공개 SOTA 모델의 faithful reproduction 및 동등 조건 비교.

합성 데이터의 ID/이름과 템플릿 계열이 반복되므로, 모델이 생성기의 기본 속성을 학습하기 쉽습니다. 현재 IoU 상승은 코드 연결·학습 가능성 검증입니다. 새로운 웹사이트에서도 된다는 증거가 아닙니다.

## 연구 주장 전에 필요한 실험

1. 실제 AR 출력 오류 데이터로 fine-tuning, site/template holdout test.
2. Oracle frames와 predicted frames를 나누고 후자는 독립 bbox 정답으로 평가.
3. 초기 AR, frame extraction, denoising, 최종 브라우저까지 end-to-end 시간을 합산.
4. AR revise, deterministic solver, equal-budget random search, tree denoiser 비교.
5. proxy/final browser의 좌표 차이·실패율을 보고하고 unsupported CSS를 숨기지 않기.
6. 공식 구현을 사용한 추가 SOTA 비교와 3개 이상 seed.

출처: [ICLR 2025 원 논문](https://proceedings.iclr.cc/paper_files/paper/2025/hash/666dd0d92a64396e753c691db93493d4-Abstract-Conference.html), [WebUI 공식 데이터 코드](https://github.com/js0nwu/webui), [Qwen3-VL 공식 모델 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
