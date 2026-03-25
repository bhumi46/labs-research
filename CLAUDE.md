# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A full MLOps learning project — House Price Predictor built on the California Housing dataset.
Designed to teach the complete MLOps lifecycle: data versioning → experiment tracking → model registry → serving → CI/CD → monitoring.

**Two-repo setup:**
- `mlops/` (this repo) — source code + Helm chart + CI/CD workflows
- `mlops-platform-gitops/` — GitOps repo consumed by PipeCD to deploy to Kubernetes

## Commands

```bash
# Setup
pip install -e ".[dev]" && pre-commit install && dvc init

# Run full pipeline (all 5 stages)
dvc repro               # or: make pipeline

# Run individual stages
python -m src.data.ingest
python -m src.data.preprocess
python -m src.features.build_features
python -m src.models.train
python -m src.models.evaluate

# Tests
pytest tests/ -v --cov=src --cov=serve
pytest tests/test_api.py -v           # single file

# Code quality
pre-commit run --all-files

# Serving
uvicorn serve.app:app --reload        # or: make serve
# Docs at: http://localhost:8000/docs

# MLflow UI
mlflow ui                             # or: make mlflow-ui
# http://localhost:5000

# Monitoring
python -m src.monitoring.data_drift   # or: make monitor

# Docker stack
docker-compose up -d                  # or: make docker-compose-up

# Metrics comparison
dvc metrics show && dvc metrics diff

# Helm chart (local validation only — publishing is done by release.yml)
helm lint charts/house-price-predictor
helm template house-price-predictor charts/house-price-predictor

# Release (triggers release.yml → Docker + Helm OCI push + GitOps update)
git tag v1.0.2 && git push origin v1.0.2
```

## Architecture

**Pipeline (DVC):** `dvc.yaml` defines a 5-stage DAG. Run `dvc dag` to visualize.

```
ingest → preprocess → build_features → train → evaluate
```

- All hyperparameters live in `params.yaml` (single source of truth). Changing a param + `dvc repro` only re-runs affected stages.
- DVC tracks data files in `data/` and `models/scaler.pkl` with content hashes.

**Experiment Tracking (MLflow):** `src/models/train.py` logs every run (params, metrics, model artifact) to `./mlruns/`. The model is auto-registered in the MLflow Model Registry. Promote with: `client.transition_model_version_stage(name, version, "Production")`.

**Serving (FastAPI):** `serve/app.py` loads the "Production" model from the MLflow Registry at startup (falls back to a local `.pkl` file if unreachable). Feature engineering inside `/predict` mirrors `build_features.py` — must stay in sync.

**Key invariant:** The scaler in `models/scaler.pkl` is fit on training data only. At inference time, `serve/app.py` replicates the same feature engineering logic manually (no scaler used server-side — raw features are sent by the client). If you change the feature list in `build_features.py`, update `FEATURE_ORDER` in `serve/app.py` to match.

**CI/CD:**
- `ci.yml` — lint → test → `dvc repro` on every push. Posts ML metrics as PR comment.
- `cd.yml` — builds Docker image (sha tag) on every PR → updates staging in GitOps repo. On merge to main → updates production in GitOps repo. PipeCD picks up changes and deploys.
- `release.yml` — triggered by `git tag vX.Y.Z`. Builds Docker (version tag) + packages Helm chart + pushes both to `ghcr.io` OCI registry + updates GitOps repo production values + creates GitHub Release.

**Helm Chart:** `charts/house-price-predictor/` — lives in this repo alongside source code (MOSIP pattern).
- `Chart.yaml` has two version fields: `version` (chart — bump only when templates change) and `appVersion` (app — auto-bumped by `release.yml` on every tag).
- Templates: `deployment.yaml`, `service.yaml`, `ingress.yaml`, `external-secret.yaml`
- Hooks: `hooks/pre-upgrade.yaml` (validates MLflow model exists before upgrade), `hooks/post-install.yaml` (smoke tests the API after deploy)
- Chart is published to OCI: `oci://ghcr.io/YOUR_USERNAME/charts/house-price-predictor`

**Secret Management (zero secrets in Git):**
- `ExternalSecret` CRD in `templates/external-secret.yaml` — tells ESO to fetch secrets from Vault and create a K8s Secret
- All secret config (Vault path, keys, refreshInterval, secretStoreName) lives in `values.yaml` under `externalSecret:` — never hardcoded in templates
- Vault path: `secret/mlops/house-price-predictor` (stores mlflow-db-password, minio-access-key, minio-secret-key)
- Disable per environment: set `externalSecret.enabled: false` in override values

**Automatic Secret Rotation (zero manual steps):**
- Stakater Reloader watches the K8s Secret created by ESO
- When Vault secret is rotated → ESO updates K8s Secret → Reloader triggers rolling restart automatically
- Reloader annotation key/value lives in `values.yaml` under `reloader:` — not hardcoded in templates
- Disable per environment: set `reloader.enabled: "false"` in override values
- Reloader must be installed in the platform: `helm install reloader stakater/reloader -n mlops-platform`

**Release flow (MOSIP pattern):**
```
develop → release/x.y.z → QA sign-off → git tag vX.Y.Z → push tag
                                                  ↓
                                          release.yml triggers:
                                          Docker push + Helm OCI push + GitOps update
```

**Monitoring:** `src/monitoring/data_drift.py` uses Evidently to compare training data distribution vs. incoming data. Outputs `reports/drift_report.html`.

## Target: All 5 DVC Stages Pass + Tests Green = Ready to Deploy
