# ADR 004: Agent Runtime Selection

## Status
Accepted

## Context
The platform requires an autonomous agent that can:
- Analyze login events and determine risk
- Call tools (risk scoring, user history lookup, quarantine)
- Make decisions based on policy
- Explain its reasoning

Key requirements:
- **Structured workflows**: Predictable state transitions
- **Tool integration**: Call Python functions as tools
- **Observability**: Log and trace agent decisions
- **Testability**: Unit test individual components

## Decision
Use **LangGraph** for agent orchestration.

### Why LangGraph
1. **Graph-based workflows** - Explicit state machine, not unbounded loops
2. **Built on LangChain** - Access to extensive tool ecosystem
3. **Typed state** - TypedDict state schema catches errors early
4. **Conditional edges** - Route based on state (e.g., risk level)
5. **Checkpointing** - Resume from failures

### Architecture
```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Analyze   │  ← Compute risk score
                    │   Event     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │ high/critical           │ low/medium
              ▼                         ▼
      ┌───────────────┐         ┌───────────────┐
      │  Investigate  │         │    Decide     │
      │  (get history,│         │   (allow)     │
      │   similar)    │         └───────┬───────┘
      └───────┬───────┘                 │
              │                         │
              ▼                         │
      ┌───────────────┐                 │
      │    Decide     │                 │
      │(block/step-up)│                 │
      └───────┬───────┘                 │
              │                         │
              └────────────┬────────────┘
                           │
                    ┌──────▼──────┐
                    │    END      │
                    └─────────────┘
```

## Implementation

### State Definition
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: Optional[str]
    risk_score: Optional[float]
    risk_level: Optional[str]
    investigation_notes: List[str]
    recommended_action: Optional[str]
```

### Graph Construction
```python
workflow = StateGraph(AgentState)

workflow.add_node("analyze", analyze_event)
workflow.add_node("investigate", investigate)
workflow.add_node("decide", decide_action)

workflow.set_entry_point("analyze")
workflow.add_conditional_edges(
    "analyze",
    should_investigate,
    {"investigate": "investigate", "decide": "decide"}
)
workflow.add_edge("investigate", "decide")
workflow.add_edge("decide", END)

agent = workflow.compile()
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **LangGraph (chosen)** | Structured, typed, observable | Newer, smaller community |
| **LangChain Agents** | Mature, well-documented | ReAct loop can be unpredictable |
| **AutoGen** | Multi-agent support | Heavy, complex setup |
| **Custom FSM** | Full control | Rebuild tool integration |

## Consequences

### Positive
- Predictable execution flow
- Easy to test individual nodes
- Clear visualization of agent logic
- Type-safe state management

### Negative
- Less flexible than ReAct agents
- Requires upfront graph design
- Learning curve for graph concepts

## MCP-like Tool Pattern
Tools follow a schema-first design inspired by MCP:

```python
TOOL_SCHEMA = {
    "name": "quarantine_account",
    "description": "Temporarily quarantine a user account",
    "input_schema": QuarantineInput.model_json_schema(),
    "output_schema": QuarantineOutput.model_json_schema(),
}

def execute(input_data: QuarantineInput) -> QuarantineOutput:
    # Implementation
    pass
```

## References
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Model Context Protocol](https://modelcontextprotocol.io/)
