# ADR 001: Feature Store Implementation

## Status

Accepted

## Context

Need consistent online/offline feature access for risk scoring across tenants.
Features include: failed_logins_24h, ip_reputation, device_age_days, location_velocity, etc.

Key requirements:
- Training/serving consistency (avoid skew)
- Low-latency online serving (<10ms)
- Feature versioning and lineage
- Multi-tenant isolation
- Local development friendly

## Decision

**Feast (local mode)**:
- Offline store: Parquet files (historical training)
- Online store: SQLite (low-latency serving for dev)
- Future: Cloud migration path to Postgres/Redis

## Alternatives Considered

| Option | Pros | Cons | Score |
|--------|------|------|-------|
| Feast | ML-native, lineage, open source | Learning curve | 9/10 |
| Redis only | Simple, fast | No versioning, no offline | 6/10 |
| Tecton | Fully managed | $$$ (not free tier) | 4/10 |
| Custom pandas | Simple | No consistency guarantees | 3/10 |

## Consequences

### Positive
- Familiar production tool (used at scale)
- Local-first development experience
- Clear migration path to cloud providers
- Feature definitions are versioned in git

### Negative
- SQLite limits throughput for prod (need Postgres migration)
- Additional learning curve for team members
- Some operational overhead

### Risks
- Online store performance may need upgrade for production traffic

## References

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store comparison](https://www.featurestore.org/)
