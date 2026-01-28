# ADR 001: Feature Store Selection

## Status
Accepted

## Context
The identity risk platform needs consistent feature computation across:
- **Training time**: Batch feature computation for model training
- **Serving time**: Real-time feature retrieval for scoring
- **Analysis**: Historical feature exploration

Without a feature store, we risk:
- Training-serving skew (different feature logic in training vs production)
- Duplicated feature computation code
- No feature versioning or lineage

## Decision
Use **Feast** as the feature store with local file backend.

### Why Feast
1. **Open source** - No vendor lock-in, active community
2. **Online/offline consistency** - Same feature definitions serve both paths
3. **Point-in-time correctness** - Prevents data leakage in training
4. **Multiple backends** - Can scale from local files to Redis/DynamoDB
5. **Python-native** - Integrates with pandas, scikit-learn

### Configuration
```yaml
# feature_store/feature_store.yaml
project: identity_risk
provider: local
online_store:
  type: sqlite
  path: data/online_store.db
offline_store:
  type: file
entity_key_serialization_version: 3
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Feast (chosen)** | OSS, flexible backends, active development | Steeper learning curve |
| **Tecton** | Managed service, enterprise features | Expensive, vendor lock-in |
| **Hopsworks** | Full ML platform | Heavy, complex setup |
| **Custom solution** | Full control | Maintenance burden, rebuild from scratch |

## Consequences

### Positive
- Feature definitions are version-controlled
- Online/offline feature parity guaranteed
- Easy migration to cloud backends (Redis, DynamoDB)
- Built-in data quality checks

### Negative
- Additional infrastructure component
- Feast version upgrades may require migrations
- Local SQLite limits concurrent access

## References
- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store for ML](https://www.featurestore.org/)
