install:
	pip install -e .[dev]
	pre-commit install

dev:
	pip install -e .[dev,docs]

build:
	rm -rf dist
	pip install build
	python -m build

.PHONY: docs
docs:
	jb build docs
