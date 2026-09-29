.PHONY: install install-browser test test-browser check smoke build

install:
	python -m pip install -e '.[dev]'

install-browser:
	python -m pip install -e '.[browser,test]'
	python -m playwright install chromium

test:
	python -m pytest -q -m 'not browser'

test-browser:
	python -m pytest -q -m browser

check:
	python -m ruff check framediff tests
	python -m pytest -q -m 'not browser'

smoke:
	python -m framediff generate --out data/smoke --count 150 --seed 42
	python -m framediff suite --config configs/smoke.json

build:
	python -m build
