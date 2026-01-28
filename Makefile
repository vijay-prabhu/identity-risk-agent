.PHONY: install dev test test-model-gate lint clean

# Install dependencies
install:
	pip install -r requirements.txt

# Start development servers (FastAPI + Streamlit)
dev:
	uvicorn api.main:app --reload --port 8000 & \
	streamlit run ui/app.py --server.port 8501

# Run API server only
api:
	uvicorn api.main:app --reload --port 8000

# Run UI only
ui:
	streamlit run ui/app.py --server.port 8501

# Run all tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

# Run model quality gates
test-model-gate:
	pytest tests/test_models/ -v -k "quality_gate"

# Run data validation tests
test-data:
	pytest tests/test_data/ -v

# Run API tests
test-api:
	pytest tests/test_api/ -v

# Lint code
lint:
	ruff check src/ api/ tests/
	ruff format --check src/ api/ tests/

# Format code
format:
	ruff format src/ api/ tests/

# MLflow UI
mlflow-ui:
	mlflow ui --port 5000

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# Generate synthetic data (Phase 1)
generate-data:
	python -c "from src.core.data_generator import generate_logins; generate_logins()"
