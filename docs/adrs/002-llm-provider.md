# ADR 002: LLM Provider Strategy

## Status
Accepted

## Context
The GenAI agent needs an LLM for:
- **Risk explanations**: Natural language explanations of scoring decisions
- **Agent reasoning**: Tool selection and action planning
- **RAG responses**: Contextual answers using retrieved documents

Key requirements:
- Low latency (< 2s for explanations)
- Privacy-safe (no PII sent to external APIs)
- Cost-effective for demo/portfolio use
- Production-ready path available

## Decision
Use a **tiered LLM strategy** with local-first approach:

### Tier 1: Local Inference (Default)
- **Ollama** with small models (Llama 3.2 3B, Mistral 7B)
- Runs entirely on developer machine
- No API costs, full privacy
- Suitable for demo and development

### Tier 2: Template Fallback
- Rule-based response generation when LLM unavailable
- Structured templates for common explanations
- Zero latency, deterministic outputs

### Tier 3: Cloud API (Optional)
- OpenAI/Anthropic for production deployments
- Higher quality but requires API keys
- PII redaction applied before sending

## Architecture
```
User Query
    │
    ▼
┌─────────────────┐
│  PII Redaction  │  ← Presidio removes sensitive data
└────────┬────────┘
         │
    ┌────▼────┐
    │ Ollama? │──No──► Template Response
    └────┬────┘
         │ Yes
         ▼
┌─────────────────┐
│  Local LLM      │  ← Llama 3.2 / Mistral
│  (Ollama)       │
└────────┬────────┘
         │
         ▼
    Response
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **OpenAI only** | Best quality, easy setup | Cost, privacy concerns, API dependency |
| **Local only (chosen)** | Privacy, no cost | Requires local GPU/CPU resources |
| **vLLM** | High throughput | Complex setup, GPU required |
| **LMStudio** | Nice UI | Not programmatic |

## Consequences

### Positive
- Zero API costs for development and demo
- Full privacy - no data leaves the machine
- Works offline
- Easy to upgrade to cloud APIs later

### Negative
- Quality varies with local model size
- Requires 8GB+ RAM for larger models
- First inference is slower (model loading)

## References
- [Ollama](https://ollama.ai/)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/llms/ollama)
