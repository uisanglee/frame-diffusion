# Contributing

FrameDiff is alpha-stage research software. Please keep claims, datasets, and
baselines explicit and reproducible.

## Development setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make install
make test
```

Browser parity tests additionally require:

```bash
make install-browser
make test-browser
```

## Pull requests

1. Open an issue for changes that alter LayoutIR semantics, evaluation protocol,
   datasets, or public CLI behavior.
2. Create a branch such as `feat/absolute-layout` or `fix/split-leakage`.
3. Add tests and update README/CHANGELOG when behavior changes.
4. Run `make check`; run browser tests when compiler or executor behavior changes.
5. Never commit datasets, checkpoints, API keys, screenshots, or experiment runs.
6. Report hardware, seed, data split, command, and raw artifact location for performance claims.

## Research integrity

- Do not call fitted WebUI surrogates original CSS ground truth.
- Keep predicted frame inputs separate from independent evaluation ground truth.
- Do not relabel internal baselines as published model reproductions.
- Report failed runs, unsupported samples, and data exclusions.
- Preserve site/domain group separation between training and evaluation.

## Commit style

Use concise imperative subjects. Conventional Commit prefixes are recommended:
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, and `chore:`.
