# ADR 003: Multi-Tenant Architecture

## Status
Accepted

## Context
Enterprise identity platforms serve multiple customers (tenants) from shared infrastructure. Each tenant requires:
- **Data isolation**: One tenant cannot access another's data
- **Configuration isolation**: Per-tenant settings (thresholds, models)
- **Resource isolation**: Fair resource allocation

The risk scoring platform must support multi-tenancy at all layers.

## Decision
Implement **logical multi-tenancy** with tenant context propagation.

### Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│  TenantMiddleware extracts tenant_id from headers/JWT   │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Context Layer                         │
│  TenantContext (contextvars) propagates tenant_id       │
└────────────────────────┬────────────────────────────────┘
                         │
     ┌───────────────────┼───────────────────┐
     ▼                   ▼                   ▼
┌─────────┐      ┌──────────────┐     ┌───────────┐
│ Feature │      │ Vector Store │     │  Model    │
│  Store  │      │   (Qdrant)   │     │ Registry  │
│ tenant  │      │  namespace   │     │  per-     │
│ filter  │      │  filtering   │     │  tenant   │
└─────────┘      └──────────────┘     └───────────┘
```

### Implementation

#### 1. API Middleware
```python
class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID", "default")
        with TenantContext(tenant_id):
            response = await call_next(request)
        return response
```

#### 2. Context Propagation
```python
# Thread-safe tenant context using contextvars
_tenant_var: ContextVar[str] = ContextVar("tenant_id", default="default")

class TenantContext:
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    def __enter__(self):
        self.token = _tenant_var.set(self.tenant_id)
        return self

    def __exit__(self, *args):
        _tenant_var.reset(self.token)
```

#### 3. Vector Store Filtering
```python
# Qdrant query with tenant filter
results = client.query_points(
    collection_name="events",
    query=embedding,
    query_filter=Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    ),
)
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Logical (chosen)** | Simple, cost-effective | Requires careful filter enforcement |
| **Physical (separate DBs)** | Strongest isolation | Expensive, complex management |
| **Schema-based** | Good isolation | Database-specific, migrations complex |

## Consequences

### Positive
- Single deployment serves all tenants
- Easy to add new tenants (just a new ID)
- Shared infrastructure reduces cost
- Consistent codebase

### Negative
- Must enforce tenant filters everywhere
- Noisy neighbor risk without rate limiting
- Cross-tenant queries require special handling

## Security Considerations

1. **Always filter by tenant_id** in all data access
2. **Validate tenant_id** in middleware
3. **Log tenant context** for audit trails
4. **Rate limit per tenant** to prevent abuse

## References
- [Multi-tenancy Patterns](https://docs.microsoft.com/en-us/azure/architecture/guide/multitenant/overview)
- [Qdrant Filtering](https://qdrant.tech/documentation/concepts/filtering/)
