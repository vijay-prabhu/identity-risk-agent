# ADR 003: Multi-Tenant Architecture

## Status

Accepted

## Context

Enterprise identity platforms must support multi-tenancy:
- Strict data isolation between tenants
- Per-tenant configuration (risk thresholds, features enabled)
- Audit logging per tenant
- API key to tenant mapping

## Decision

**Application-level multi-tenancy with context propagation**:
- Tenant ID in all data keys and queries
- FastAPI middleware for tenant extraction
- Context variables for tenant propagation
- Per-tenant configuration via TenantConfig class

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│ API Request                                               │
│   ├── X-API-Key: sk_tenant_a_xxx                         │
│   └── X-Tenant-ID: tenant_a (fallback)                   │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ TenantMiddleware                                          │
│   ├── Extract tenant from API key                        │
│   ├── Set tenant in context                              │
│   └── Add to request.state                               │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│ Feature Store / Model Serving                             │
│   ├── Filter by tenant_id                                │
│   ├── Apply tenant-specific config                       │
│   └── Audit log with tenant context                      │
└──────────────────────────────────────────────────────────┘
```

## Alternatives Considered

| Option | Pros | Cons | Score |
|--------|------|------|-------|
| App-level (chosen) | Simple, flexible | Requires discipline | 9/10 |
| Database per tenant | Strong isolation | Operational overhead | 6/10 |
| Schema per tenant | Good isolation | Migration complexity | 7/10 |

## Consequences

### Positive
- No infrastructure overhead per tenant
- Easy to add new tenants
- Consistent API across tenants
- Flexible per-tenant configuration

### Negative
- Requires careful coding to maintain isolation
- Single database = shared failure domain
- Need comprehensive testing for isolation

## Implementation

Key components in `src/core/multi_tenant.py`:
- `TenantMiddleware`: FastAPI middleware
- `TenantContext`: Context manager for tenant scope
- `TenantConfig`: Per-tenant configuration
- `TenantAuditLogger`: Audit logging

## References

- [Multi-Tenant Architecture Patterns](https://docs.microsoft.com/en-us/azure/architecture/guide/multitenant/)
