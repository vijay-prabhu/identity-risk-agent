"""
RAG Pipeline for Risk Explanations

Retrieval-Augmented Generation pipeline for explaining risk decisions
using relevant historical context.
"""

import logging
from typing import List, Optional, Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.agents.vector_store import IdentityVectorStore, create_event_text

logger = logging.getLogger(__name__)

# System prompt for risk explanation
SYSTEM_PROMPT = """You are an identity security analyst AI assistant. Your role is to explain risk scoring decisions for login events.

You have access to:
1. The current login event details
2. Similar historical events for context
3. Risk factors and their contributions

Guidelines:
- Be concise and factual
- Explain WHY the risk score was assigned
- Reference similar past events when relevant
- Suggest appropriate actions based on risk level
- Never reveal sensitive user data beyond what's necessary

Risk Level Actions:
- LOW (0-0.3): Allow login, no action needed
- MEDIUM (0.3-0.6): Monitor session, apply rate limiting
- HIGH (0.6-0.8): Require step-up authentication
- CRITICAL (0.8-1.0): Block and alert security team
"""

USER_PROMPT = """Explain the risk assessment for this login event:

Current Event:
{event_details}

Risk Score: {risk_score}
Risk Level: {risk_level}
Risk Factors: {risk_factors}

Similar Historical Events:
{context}

User Question: {query}

Provide a clear, concise explanation."""


class RiskExplainer:
    """
    RAG-powered risk explanation generator.

    Retrieves relevant context from vector store and uses LLM
    to generate human-readable explanations.
    """

    def __init__(
        self,
        vector_store: Optional[IdentityVectorStore] = None,
        llm=None,
    ):
        """
        Initialize the explainer.

        Args:
            vector_store: Vector store for context retrieval
            llm: LangChain-compatible LLM (default: mock for testing)
        """
        self.vector_store = vector_store or IdentityVectorStore()
        self.llm = llm

        # Build prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ])

    def _format_event_details(self, event: Dict[str, Any]) -> str:
        """Format event details for the prompt."""
        lines = []

        if event.get("user_id"):
            lines.append(f"- User: {event['user_id']}")
        if event.get("device_id"):
            lines.append(f"- Device: {event['device_id']}")
        if event.get("ip"):
            lines.append(f"- IP: {event['ip']}")
        if event.get("location_country"):
            lines.append(f"- Location: {event.get('location_city', '')}, {event['location_country']}")
        if event.get("timestamp"):
            lines.append(f"- Time: {event['timestamp']}")

        lines.append(f"- MFA Used: {'Yes' if event.get('mfa_used') else 'No'}")
        lines.append(f"- VPN Detected: {'Yes' if event.get('vpn_detected') else 'No'}")
        lines.append(f"- Login Success: {'Yes' if event.get('success') else 'No'}")

        return "\n".join(lines)

    def _format_context(self, similar_events: List[Dict[str, Any]]) -> str:
        """Format similar events as context."""
        if not similar_events:
            return "No similar historical events found."

        lines = []
        for i, event in enumerate(similar_events, 1):
            lines.append(f"{i}. {event.get('text', 'Unknown event')} (similarity: {event.get('score', 0):.2f})")

        return "\n".join(lines)

    def explain(
        self,
        event: Dict[str, Any],
        risk_score: float,
        risk_level: str,
        risk_factors: List[str],
        query: str = "Why was this login flagged?",
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate an explanation for a risk decision.

        Args:
            event: Login event details
            risk_score: Computed risk score (0-1)
            risk_level: Risk level (low/medium/high/critical)
            risk_factors: List of contributing factors
            query: User's question
            tenant_id: Tenant for context filtering

        Returns:
            Dict with explanation and metadata
        """
        # Create event text for similarity search
        event_text = create_event_text({
            **event,
            "risk_score": risk_score,
        })

        # Retrieve similar events
        similar_events = self.vector_store.search(
            query=event_text,
            tenant_id=tenant_id,
            limit=3,
            score_threshold=0.3,
        )

        # Format inputs
        event_details = self._format_event_details(event)
        context = self._format_context(similar_events)
        factors_str = ", ".join(risk_factors) if risk_factors else "None identified"

        # Generate explanation
        if self.llm:
            # Use LLM for generation
            chain = self.prompt | self.llm | StrOutputParser()
            explanation = chain.invoke({
                "event_details": event_details,
                "risk_score": f"{risk_score:.2f}",
                "risk_level": risk_level.upper(),
                "risk_factors": factors_str,
                "context": context,
                "query": query,
            })
        else:
            # Fallback: template-based explanation
            explanation = self._generate_template_explanation(
                event, risk_score, risk_level, risk_factors, similar_events
            )

        return {
            "explanation": explanation,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "factors": risk_factors,
            "similar_events": len(similar_events),
            "sources": [e.get("event_id", "unknown") for e in similar_events],
        }

    def _generate_template_explanation(
        self,
        event: Dict[str, Any],
        risk_score: float,
        risk_level: str,
        risk_factors: List[str],
        similar_events: List[Dict[str, Any]],
    ) -> str:
        """Generate a template-based explanation (fallback when no LLM)."""
        parts = []

        # Risk summary
        parts.append(f"This login was assessed as **{risk_level.upper()}** risk (score: {risk_score:.2f}).")

        # Explain factors
        if risk_factors:
            parts.append("\n\n**Contributing factors:**")
            for factor in risk_factors:
                parts.append(f"- {factor}")

        # Recommendations
        parts.append("\n\n**Recommended action:**")
        if risk_level == "critical":
            parts.append("Block this login attempt and alert the security team immediately. This pattern matches known attack signatures.")
        elif risk_level == "high":
            parts.append("Require additional verification (step-up MFA) before allowing access. Monitor the session closely.")
        elif risk_level == "medium":
            parts.append("Allow login but apply enhanced monitoring. Consider rate limiting API access.")
        else:
            parts.append("No action required. This login matches normal user behavior patterns.")

        # Historical context
        if similar_events:
            parts.append(f"\n\n**Historical context:** Found {len(similar_events)} similar events in the system.")

        return "".join(parts)


class RAGPipeline:
    """
    Complete RAG pipeline for the identity risk platform.

    Handles document ingestion, retrieval, and generation.
    """

    def __init__(
        self,
        vector_store: Optional[IdentityVectorStore] = None,
        llm=None,
    ):
        self.vector_store = vector_store or IdentityVectorStore()
        self.explainer = RiskExplainer(vector_store=self.vector_store, llm=llm)

    def ingest_events(
        self,
        events: List[Dict[str, Any]],
        tenant_id: str = "default",
    ) -> int:
        """
        Ingest login events into the vector store.

        Args:
            events: List of event dictionaries
            tenant_id: Tenant identifier

        Returns:
            Number of events ingested
        """
        processed = []
        for event in events:
            text = create_event_text(event)
            processed.append({
                "event_id": event.get("event_id", f"evt_{hash(str(event))}"),
                "text": text,
                **event,
            })

        self.vector_store.add_events_batch(processed, tenant_id=tenant_id)
        return len(processed)

    def query(
        self,
        query: str,
        event: Optional[Dict[str, Any]] = None,
        risk_score: float = 0.0,
        risk_level: str = "low",
        risk_factors: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG pipeline for an explanation.

        Args:
            query: User's question
            event: Current event context (optional)
            risk_score: Current risk score
            risk_level: Current risk level
            risk_factors: Contributing factors
            tenant_id: Tenant for filtering

        Returns:
            Explanation response
        """
        if event:
            return self.explainer.explain(
                event=event,
                risk_score=risk_score,
                risk_level=risk_level,
                risk_factors=risk_factors or [],
                query=query,
                tenant_id=tenant_id,
            )
        else:
            # Search-only mode
            results = self.vector_store.search(
                query=query,
                tenant_id=tenant_id,
                limit=5,
            )
            return {
                "explanation": f"Found {len(results)} relevant events.",
                "results": results,
                "sources": [r.get("event_id") for r in results],
            }


if __name__ == "__main__":
    # Quick test
    print("Testing RAG pipeline...")

    pipeline = RAGPipeline()

    # Ingest some test events
    test_events = [
        {
            "event_id": "evt_001",
            "user_id": "user_001",
            "device_id": "laptop_001",
            "location_country": "US",
            "mfa_used": True,
            "vpn_detected": False,
            "success": True,
            "risk_score": 0.1,
        },
        {
            "event_id": "evt_002",
            "user_id": "user_002",
            "device_id": "device_unknown_123",
            "location_country": "RU",
            "mfa_used": False,
            "vpn_detected": True,
            "success": False,
            "risk_score": 0.9,
        },
    ]

    count = pipeline.ingest_events(test_events)
    print(f"Ingested {count} events")

    # Query for explanation
    result = pipeline.query(
        query="Why was this login blocked?",
        event={
            "user_id": "attacker",
            "device_id": "device_unknown_999",
            "location_country": "CN",
            "mfa_used": False,
            "vpn_detected": True,
        },
        risk_score=0.95,
        risk_level="critical",
        risk_factors=["Unknown device", "VPN detected", "No MFA", "High-risk location"],
    )

    print(f"\nExplanation:\n{result['explanation']}")
