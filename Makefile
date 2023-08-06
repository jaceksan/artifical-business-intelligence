.PHONY: dev
dev: infra
	python3.11 -m venv .venv --upgrade-deps
	.venv/bin/pip3 install -r requirements.txt

.PHONY: infra
infra:
	python3.11 -m venv .venv --upgrade-deps
	.venv/bin/pip3 install -r infra-requirements.txt
	.venv/bin/pre-commit install

.PHONY: fix-all
fix-all:
	.venv/bin/pre-commit run --all-files