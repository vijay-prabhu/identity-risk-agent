# Contributing to Identity Risk Agent

Thank you for your interest in contributing!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/vijay-prabhu/identity-risk-agent
cd identity-risk-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install dev tools
pip install ruff black isort pytest-cov
```

## Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Quality gate (model AUC check)
make test-gate
```

## Code Style

We use automated formatters:

```bash
# Format code
make format

# Check style
make lint
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Format code (`make format`)
6. Commit with clear message
7. Push and create PR

## Architecture Decision Records

For significant changes, create an ADR in `docs/adrs/`:

```markdown
# ADR XXX: Title

## Status
Proposed

## Context
[Why is this change needed?]

## Decision
[What did you decide?]

## Consequences
[What are the trade-offs?]
```

## Questions?

Open an issue or reach out to the maintainers.
