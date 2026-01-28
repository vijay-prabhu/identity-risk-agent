# Identity Risk Agent Platform

> Portfolio project demonstrating a full ML/GenAI workflow for identity risk scoring - from data ingestion through feature engineering, model training, RAG-powered agents, and production deployment.

[![Tests](https://github.com/yourusername/identity-risk-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/identity-risk-agent/actions)

## Overview

This project mirrors enterprise ML platform patterns (similar to Okta's Intelligence Accelerator), showcasing:

- **ML Infrastructure**: Feature stores, MLflow tracking, distributed training
- **GenAI/Agents**: RAG pipelines, LangGraph agents, MCP-like tool protocols
- **Production Patterns**: Multi-tenant isolation, privacy/security, CI/CD, monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Identity Risk Agent Platform                  │
├─────────────────────────────────────────────────────────────────────┤
│  Data Layer          │  ML Layer           │  Agent Layer           │
│  ─────────────────   │  ─────────────────  │  ─────────────────     │
│  • Synthetic events  │  • Feature Store    │  • RAG Pipeline        │
│  • Feature pipeline  │  • Risk Model       │  • LangGraph Agents    │
│  • Vector embeddings │  • MLflow tracking  │  • MCP Tools           │
├─────────────────────────────────────────────────────────────────────┤
│  API Layer           │  Infrastructure     │  Extensions            │
│  ─────────────────   │  ─────────────────  │  ─────────────────     │
│  • FastAPI serving   │  • Docker Compose   │  • Multi-tenant        │
│  • Streamlit UI      │  • GitHub Actions   │  • Privacy (Presidio)  │
│  • Health/Score APIs │  • Monitoring       │  • Real external data  │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Phases

| Phase | Milestone | Status |
|-------|-----------|--------|
| **Phase 1** | MVP Scoring API | 🔲 Not Started |
| **Phase 2** | ML Infra Layer (Feature Store, MLflow) | 🔲 Not Started |
| **Phase 3** | GenAI Agent (RAG, LangGraph, MCP) | 🔲 Not Started |
| **Phase 4** | Production Polish (CI/CD, Monitoring) | 🔲 Not Started |
| **Phase 5** | Portfolio Ready (ADRs, Docs, Demo) | 🔲 Not Started |

## Tech Stack

**Core:**
- Data: pandas, Faker, Parquet/SQLite
- ML: scikit-learn, sentence-transformers
- Features: Feast (local mode)
- Serving: FastAPI, Streamlit
- LLM/Agents: Ollama (local), LangGraph
- Vector DB: Qdrant Cloud (free tier)
- Tracking: MLflow (local)

**Infrastructure:**
- Orchestration: Airflow (Docker)
- CI/CD: GitHub Actions
- Monitoring: Prometheus/Grafana
- Containers: Docker Compose

## Quick Start

```bash
# Clone & install
git clone https://github.com/yourusername/identity-risk-agent.git
cd identity-risk-agent
pip install -r requirements.txt

# Local dev (Phase 1)
make dev          # Starts FastAPI + Streamlit

# Run tests
make test         # Full test suite
make test-model-gate  # Model quality gates

# Full stack (Phase 4)
docker-compose up # Airflow + monitoring
```

## Project Structure

```
identity-risk-agent/
├── README.md                 # This file
├── docs/
│   └── adrs/                 # Architecture Decision Records
├── data/                     # Synthetic data + schemas
├── notebooks/                # Phase-wise Jupyter notebooks
├── src/
│   ├── core/                 # MVP scoring logic
│   ├── features/             # Feature engineering
│   ├── models/               # Training/evaluation
│   ├── agents/               # RAG + LangGraph
│   ├── tools/                # MCP-like APIs
│   ├── privacy/              # PII detection/redaction
│   └── infra/                # Docker, k8s manifests
├── api/                      # FastAPI application
├── ui/                       # Streamlit dashboard
├── tests/                    # Test suite
│   ├── test_data/            # Schema + distribution tests
│   ├── test_models/          # Quality gates (AUC > 0.85)
│   ├── test_api/             # Contract + integration
│   └── test_extensions/      # Tenant isolation, privacy
├── docker-compose.yml        # Local stack
├── .github/workflows/        # CI/CD pipelines
├── Makefile                  # Dev commands
└── requirements.txt          # Python dependencies
```

## Success Metrics

| Metric | Target |
|--------|--------|
| Live demo URL | Publicly accessible |
| Test coverage | 80%+ |
| Model quality | ROC-AUC > 0.85 |
| ADRs written | 5+ decisions |
| Extensions | 4 togglable features |

## License

MIT
