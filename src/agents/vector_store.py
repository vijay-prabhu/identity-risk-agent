"""
Vector Store for RAG Pipeline

Uses Qdrant for storing and retrieving embeddings of login events
and their explanations.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default embedding model - small and fast
DEFAULT_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "login_events"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


class IdentityVectorStore:
    """
    Vector store for identity risk events and explanations.

    Uses Qdrant for vector storage and sentence-transformers for embeddings.
    Supports multi-tenant isolation via metadata filtering.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = DEFAULT_MODEL,
        qdrant_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model name
            qdrant_path: Path for local Qdrant storage (default: in-memory)
            qdrant_url: URL for remote Qdrant server
        """
        self.collection_name = collection_name

        # Initialize Qdrant client
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        elif qdrant_path:
            self.client = QdrantClient(path=qdrant_path)
        else:
            # In-memory for development
            self.client = QdrantClient(":memory:")

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = embedding_model
        self.embedder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedder.encode(text).tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return self.embedder.encode(texts).tolist()

    def add_event(
        self,
        event: Dict[str, Any],
        tenant_id: str = "default",
    ) -> None:
        """
        Add a login event to the vector store.

        Args:
            event: Event dict with 'event_id', 'text', and optional metadata
            tenant_id: Tenant identifier for isolation
        """
        event_id = event.get("event_id", str(hash(str(event))))
        text = event.get("text", create_event_text(event))

        embedding = self.embed_text(text)

        point = PointStruct(
            id=hash(event_id) % (2**63),  # Convert to int ID
            vector=embedding,
            payload={
                "event_id": event_id,
                "text": text,
                "tenant_id": tenant_id,
                **{k: v for k, v in event.items() if k not in ["event_id", "text"]},
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def add_events_batch(
        self,
        events: List[Dict[str, Any]],
        tenant_id: str = "default",
    ) -> None:
        """
        Add multiple events in batch.

        Args:
            events: List of event dicts with 'event_id', 'text', and metadata
            tenant_id: Tenant identifier
        """
        texts = [e["text"] for e in events]
        embeddings = self.embed_texts(texts)

        points = []
        for event, embedding in zip(events, embeddings):
            point = PointStruct(
                id=hash(event["event_id"]) % (2**63),
                vector=embedding,
                payload={
                    "tenant_id": tenant_id,
                    **event,
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Added {len(points)} events to vector store")

    def search(
        self,
        query: str,
        tenant_id: Optional[str] = None,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar events.

        Args:
            query: Search query text
            tenant_id: Filter by tenant (None for all tenants)
            limit: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            List of matching events with scores
        """
        query_embedding = self.embed_text(query)

        # Build filter for tenant isolation
        query_filter = None
        if tenant_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    )
                ]
            )

        # Use query_points for newer Qdrant client API
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in results.points
        ]

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific event by ID."""
        point_id = hash(event_id) % (2**63)

        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
            )
            if points:
                return points[0].payload
        except Exception:
            pass

        return None

    def delete_event(self, event_id: str) -> None:
        """Delete an event from the store."""
        point_id = hash(event_id) % (2**63)
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=[point_id],
        )

    def count(self, tenant_id: Optional[str] = None) -> int:
        """Count events in the store."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count


def create_event_text(event: Dict[str, Any]) -> str:
    """
    Create a text description of a login event for embedding.

    Args:
        event: Event dictionary with login details

    Returns:
        Text description suitable for embedding
    """
    parts = []

    # User and device
    parts.append(f"User {event.get('user_id', 'unknown')} login")

    if event.get('device_id'):
        if 'unknown' in event['device_id'].lower():
            parts.append("from unknown device")
        else:
            parts.append(f"from device {event['device_id']}")

    # Location
    if event.get('location_country'):
        parts.append(f"in {event['location_country']}")

    # Security factors
    factors = []
    if event.get('vpn_detected'):
        factors.append("VPN detected")
    if not event.get('mfa_used'):
        factors.append("no MFA")
    if event.get('is_new_device'):
        factors.append("new device")
    if not event.get('success'):
        factors.append("failed attempt")

    if factors:
        parts.append(f"with {', '.join(factors)}")

    # Risk level
    if event.get('risk_score') is not None:
        score = event['risk_score']
        if score >= 0.8:
            parts.append("- CRITICAL risk")
        elif score >= 0.6:
            parts.append("- HIGH risk")
        elif score >= 0.3:
            parts.append("- MEDIUM risk")
        else:
            parts.append("- LOW risk")

    return " ".join(parts)


if __name__ == "__main__":
    # Quick test
    print("Testing vector store...")

    store = IdentityVectorStore()

    # Add test events
    test_events = [
        {
            "event_id": "evt_001",
            "text": "User login from known device in US with MFA - low risk",
            "user_id": "user_001",
            "risk_score": 0.1,
        },
        {
            "event_id": "evt_002",
            "text": "User login from unknown device in Russia with VPN no MFA - critical risk",
            "user_id": "user_002",
            "risk_score": 0.95,
        },
    ]

    store.add_events_batch(test_events, tenant_id="test")

    # Search
    results = store.search("suspicious login VPN", tenant_id="test", limit=2)
    print(f"\nSearch results for 'suspicious login VPN':")
    for r in results:
        print(f"  - {r['event_id']}: {r['text'][:50]}... (score: {r['score']:.3f})")

    print(f"\nTotal events: {store.count()}")
