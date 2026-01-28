# ADR 002: LLM Provider Selection

## Status

Accepted

## Context

The identity risk platform needs LLM capabilities for:
- RAG-powered explanations of risk decisions
- Autonomous agents for security investigations
- Natural language interfaces for analysts

Key requirements:
- Low latency for interactive use (<2s)
- Cost-effective for high volume
- Privacy-compliant (no data leaving tenant boundaries)
- Local development friendly

## Decision

**Ollama (local) as primary, with cloud fallback**:
- Local: Ollama with Llama3/Mistral for development and privacy-sensitive tenants
- Cloud: OpenAI/Anthropic API for production (with data anonymization)

## Alternatives Considered

| Option | Pros | Cons | Score |
|--------|------|------|-------|
| Ollama (local) | Privacy, free, no network | Slower, limited context | 8/10 |
| OpenAI API | Fast, high quality | Cost, data privacy | 7/10 |
| Self-hosted LLM | Full control | Complex ops, expensive | 5/10 |
| No LLM | Simpler | Missing key features | 3/10 |

## Consequences

### Positive
- Privacy-first: Local LLM option for sensitive tenants
- Cost control: No API costs during development
- Flexibility: Can switch providers without code changes

### Negative
- Local LLM requires more compute resources
- Quality varies between providers
- Need to maintain abstraction layer

## Implementation

```python
# src/agents/llm_provider.py
class LLMProvider:
    def __init__(self, provider: str = "ollama"):
        if provider == "ollama":
            self.client = OllamaClient()
        elif provider == "openai":
            self.client = OpenAIClient()
```

## References

- [Ollama Documentation](https://ollama.ai/)
- [LangChain LLM Providers](https://python.langchain.com/docs/integrations/llms/)
