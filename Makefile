install:
	pip install -e .[dev]
	pre-commit install

dev:
	pip install -e .[dev,docs]

.PHONY: docs
docs:
	jb build docs
