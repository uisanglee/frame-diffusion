# GitHub 저장소 운영 가이드

## 최초 게시

공식 공개 저장소는 <https://github.com/uisanglee/frame-diffusion>입니다. 아래 절차는 새 fork나
이전 설치를 다시 게시할 때의 참고용입니다.

현재 로컬 저장소의 기본 branch는 `main`입니다. GitHub에서 빈 저장소를 만들 때 README,
license, `.gitignore`를 추가하지 마세요. 이미 로컬에 모두 있습니다.

GitHub CLI를 사용할 경우:

```bash
git add .
git commit -m "feat: initialize FrameDiff research toolkit"
gh repo create frame-diffusion --private --source=. --remote=origin --push
```

공개 저장소로 바로 만들려면 `--private` 대신 `--public`을 사용합니다. 공개 논문이나 기관
프로젝트로 전환할 때는 `CITATION.cff`의 실제 저자·기관을 추가하세요.

CLI 없이 웹에서 저장소를 만든 경우:

```bash
git add .
git commit -m "feat: initialize FrameDiff research toolkit"
git remote add origin git@github.com:OWNER/frame-diffusion.git
git push -u origin main
```

이 문서의 명령은 예시이며 기존 공식 원격에 다시 적용하지 마세요.

## 권장 저장소 설정

GitHub의 Settings에서 다음을 켜세요.

- Actions 권한은 read-only 기본값, 승인되지 않은 fork workflow는 수동 승인.
- `main` branch rule: pull request 필수, CI의 세 job 필수, force push/delete 금지.
- Dependabot alerts와 security updates.
- Private vulnerability reporting.
- 공개 연구 코드라면 Discussions와 issue labels `experiment`, `dataset`, `reproduction`.

혼자 개발하는 초기 단계에서는 PR 승인 인원을 0으로 두되 CI 통과는 유지할 수 있습니다.

## 버전과 릴리스

사용자에게 영향을 주는 변경은 `CHANGELOG.md`의 `[Unreleased]`에 기록합니다. 릴리스할 때:

1. `pyproject.toml`, `framediff/__init__.py`, `CITATION.cff` 버전을 함께 변경.
2. `[Unreleased]` 내용을 날짜가 있는 버전 항목으로 이동.
3. CI 통과 후 `v0.1.1` 같은 annotated tag와 GitHub Release 생성.
4. checkpoint를 공개할 경우 Git 저장소에 직접 넣지 말고 Release asset이나 모델 저장소를 사용.

## 데이터와 실험 산출물

다음은 `.gitignore` 대상입니다: `data/`, `runs/`, `.cache/`, `.venv/`, model checkpoint,
Weights & Biases/MLflow 로컬 기록. 원본 데이터는 라이선스 때문에 Git에 복제하지 않습니다.

재현성을 위해 PR/issue에는 다음만 기록합니다.

- dataset 이름, 공식 URL, version/hash, split 생성 규칙
- commit SHA, config 파일, seed, 명령, hardware/software
- 원시 결과 artifact 위치와 요약표

소형 공개 fixture가 필요하면 개인정보·라이선스를 검토한 뒤 `tests/fixtures/`에 넣고 출처를
명시합니다. 대형 파일은 Git LFS로 무조건 넣기보다 dataset/model registry를 우선합니다.

## 브랜치와 리뷰

- 기능: `feat/...`, 버그: `fix/...`, 실험: `exp/...`, 문서: `docs/...`
- 한 PR에는 한 논리적 변경만 포함.
- LayoutIR 스키마 변경은 migration/호환성 설명과 테스트가 필수.
- 성능 PR은 최소 3 seeds의 평균·표준편차와 같은 예산의 baseline을 포함.
- 자동 생성 결과는 source와 분리하고 사람 검토 없이 논문 수치로 사용하지 않음.
