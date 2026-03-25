.PHONY: setup install lint test pipeline ingest preprocess features train evaluate \
        serve mlflow-ui monitor metrics dag docker-build docker-run \
        docker-compose-up docker-compose-down clean

# ── Setup ──────────────────────────────────────────────────────────────────────
setup: install
	pre-commit install
	dvc init --no-scm || dvc init  # --no-scm if git not initialized yet

install:
	pip install -e ".[dev]"

# ── Code Quality ───────────────────────────────────────────────────────────────
lint:
	pre-commit run --all-files

# ── Testing ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=src --cov=serve --cov-report=term-missing

# Run a single test file: make test-one FILE=tests/test_data.py
test-one:
	pytest $(FILE) -v

# ── DVC Pipeline ───────────────────────────────────────────────────────────────
pipeline:
	dvc repro

# Run individual stages
ingest:
	python -m src.data.ingest

preprocess:
	python -m src.data.preprocess

features:
	python -m src.features.build_features

train:
	python -m src.models.train

evaluate:
	python -m src.models.evaluate

# ── MLflow ─────────────────────────────────────────────────────────────────────
mlflow-ui:
	@echo "MLflow UI → http://localhost:5000"
	mlflow ui --host 0.0.0.0 --port 5000

# ── Serving ────────────────────────────────────────────────────────────────────
serve:
	@echo "API docs → http://localhost:8000/docs"
	uvicorn serve.app:app --reload --host 0.0.0.0 --port 8000

# ── Monitoring ─────────────────────────────────────────────────────────────────
monitor:
	python -m src.monitoring.data_drift
	@echo "Drift report → reports/drift_report.html"

# ── Metrics ────────────────────────────────────────────────────────────────────
metrics:
	dvc metrics show
	@echo "---"
	dvc metrics diff

dag:
	dvc dag

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-build:
	docker build -f docker/Dockerfile.serve -t house-price-predictor:latest .

docker-run:
	docker run -p 8000:8000 \
	  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
	  house-price-predictor:latest

docker-compose-up:
	@echo "Starting MLflow (localhost:5000) + API (localhost:8000)..."
	docker-compose up -d
	@echo "MLflow UI → http://localhost:5000"
	@echo "API docs  → http://localhost:8000/docs"

docker-compose-down:
	docker-compose down

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/raw/ data/processed/ models/ metrics/ reports/ mlruns/ mlartifacts/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".coverage" -delete
