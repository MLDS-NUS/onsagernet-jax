.PHONY: tests docs dependencies env


dependencies:
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

docs:
	@echo "Save documentation to docs..."
	pdoc onsagernet -o docs -d google --math