.PHONY: all setup setup-genai data analysis genai kernel test clean help
.DEFAULT_GOAL := help

PYTHON := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

all: data analysis ## Download data + run the full analysis notebook

setup: ## Create venv and install pinned dependencies
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

setup-genai: ## Install Phase 4 gen AI extras (needs OpenAI key)
	.venv/bin/python -m pip install openai==1.54.4 instructor==1.6.4 pydantic==2.9.2 python-dotenv==1.0.1 shap==0.46.0

kernel: ## Register the venv as a Jupyter kernel
	.venv/bin/python -m ipykernel install --user --name readmission-eval --display-name "Python (readmission-eval)"

data: ## Download + cache the UCI dataset (executes data_prep.ipynb)
	$(PYTHON) -m papermill data_prep.ipynb /tmp/data_prep_out.ipynb

analysis: ## Execute the main analysis notebook headlessly (Phases 1-3)
	$(PYTHON) -m papermill readmission_analysis.ipynb readmission_analysis.ipynb

genai: ## Execute the gen AI audit notebook (Phase 4, needs OpenAI key)
	$(PYTHON) -m papermill genai_audit.ipynb genai_audit.ipynb

test: ## Run the integrity test suite (leakage / cohort / split guards)
	$(PYTHON) -m pytest -q

clean: ## Remove caches (keeps downloaded data)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
