install:
	pip install -e .[dev,docs]
	pre-commit install

.PHONY: docs
docs:
	jb build docs
