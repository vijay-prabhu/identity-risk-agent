# Identity Risk Agent - Makefile
# Common commands for development and deployment

.PHONY: help install dev test lint format docker-build docker-up docker-down clean

# Default target
help:
	@echo "Identity Risk Agent - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Development:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Run API + Streamlit UI locally"
	@echo "  make api         - Run API server only"
	@echo "  make ui          - Run Streamlit UI only"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo "  make test-cov    - Run tests with coverage"
	@echo "  make test-gate   - Run model quality gate"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint        - Run linters (ruff, black, isort)"
	@echo "  make format      - Auto-format code"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start full stack (docker-compose)"
	@echo "  make docker-down  - Stop full stack"
	@echo ""
	@echo "Data & Model:"
	@echo "  make data        - Generate synthetic data"
	@echo "  make train       - Train risk model"
	@echo "  make features    - Update feature store"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove build artifacts"

# =============================================================================
# Installation
# =============================================================================

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "Installation complete!"

install-dev: install
	pip install ruff black isort pytest pytest-cov pytest-xdist bandit safety
	@echo "Dev dependencies installed!"

# =============================================================================
# Development
# =============================================================================

dev:
	@echo "Starting API and UI..."
	@trap 'kill %1 %2 2>/dev/null' EXIT; \
	uvicorn api.main:app --reload --port 8000 & \
	streamlit run ui/app.py --server.port 8501 & \
	wait

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run ui/app.py --server.port 8501

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=src --cov=api --cov-report=term-missing --cov-report=html

test-gate:
	@echo "Running model quality gate..."
	@python -c "from src.models.risk_model import train_and_evaluate; from src.features.feature_engineering import engineer_features; import pandas as pd; df = pd.read_parquet('data/login_events.parquet'); df = engineer_features(df); results = train_and_evaluate(df); auc = results['auc']; print(f'Model AUC: {auc:.4f}'); assert auc >= 0.85, f'Quality gate FAILED: AUC {auc:.4f} < 0.85'; print('Quality gate PASSED!')"

test-fast:
	pytest tests/ -v --tb=short -x -q

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "Running ruff..."
	-ruff check src/ api/ tests/
	@echo "\nRunning black check..."
	-black --check src/ api/ tests/
	@echo "\nRunning isort check..."
	-isort --check-only src/ api/ tests/

format:
	@echo "Formatting with black..."
	black src/ api/ tests/
	@echo "Sorting imports with isort..."
	isort src/ api/ tests/
	@echo "Fixing with ruff..."
	ruff check --fix src/ api/ tests/

security:
	@echo "Running bandit security scan..."
	-bandit -r src/ api/ -ll
	@echo "\nChecking dependencies..."
	-safety check -r requirements.txt

# =============================================================================
# Docker
# =============================================================================

docker-build:
	docker build -t identity-risk-agent:latest .
	docker build -t identity-risk-agent-ui:latest -f Dockerfile.streamlit .

docker-up:
	docker-compose up -d
	@echo ""
	@echo "Services started:"
	@echo "  - API:      http://localhost:8000"
	@echo "  - UI:       http://localhost:8501"
	@echo "  - MLflow:   http://localhost:5000"
	@echo "  - Airflow:  http://localhost:8080"
	@echo "  - Grafana:  http://localhost:3000"
	@echo "  - Qdrant:   http://localhost:6333"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Data & Model
# =============================================================================

data:
	@echo "Generating synthetic login events..."
	@python -c "from src.core.data_generator import generate_login_events; df = generate_login_events(n_events=10000, fraud_rate=0.1); df.to_parquet('data/login_events.parquet', index=False); print(f'Generated {len(df)} events -> data/login_events.parquet')"

train:
	@echo "Training risk model..."
	@python -c "from src.models.risk_model import train_and_evaluate; from src.features.feature_engineering import engineer_features; import pandas as pd; df = pd.read_parquet('data/login_events.parquet'); df = engineer_features(df); results = train_and_evaluate(df); results['scorer'].save('models/risk_model.pkl'); print(f'Model trained - AUC: {results[\"auc\"]:.4f}'); print('Saved to models/risk_model.pkl')"

features:
	@echo "Materializing features to online store..."
	cd feature_store && feast apply && feast materialize-incremental $$(date -u +%Y-%m-%dT%H:%M:%S)

# =============================================================================
# Cleanup
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml 2>/dev/null || true
	@echo "Cleaned!"

clean-data:
	rm -rf data/staging/* data/training/* 2>/dev/null || true
	@echo "Cleaned staging data!"

# =============================================================================
# Shortcuts
# =============================================================================

t: test
f: format
l: lint
d: dev
