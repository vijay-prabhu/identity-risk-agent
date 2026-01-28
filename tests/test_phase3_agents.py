"""
Tests for Phase 3: GenAI Agent components.

Tests vector store, RAG pipeline, LangGraph agent, and MCP-like tools.
"""

import pytest
from datetime import datetime


# =============================================================================
# Vector Store Tests
# =============================================================================

class TestVectorStore:
    """Tests for the identity vector store."""

    def test_vector_store_initialization(self):
        """Test vector store initializes correctly."""
        from src.agents.vector_store import IdentityVectorStore

        store = IdentityVectorStore()
        assert store is not None
        assert store.embedding_model is not None

    def test_add_single_event(self):
        """Test adding a single event."""
        from src.agents.vector_store import IdentityVectorStore

        store = IdentityVectorStore()

        event = {
            "event_id": "test_001",
            "text": "Login from known device in San Francisco",
            "user_id": "user_001",
            "risk_score": 0.1,
        }

        # Should not raise
        store.add_event(event, tenant_id="test_tenant")

    def test_add_events_batch(self):
        """Test batch event ingestion."""
        from src.agents.vector_store import IdentityVectorStore

        store = IdentityVectorStore()

        events = [
            {
                "event_id": f"batch_{i}",
                "text": f"Test event {i}",
                "user_id": f"user_{i}",
            }
            for i in range(5)
        ]

        store.add_events_batch(events, tenant_id="test_tenant")

    def test_search_returns_results(self):
        """Test search returns relevant results."""
        from src.agents.vector_store import IdentityVectorStore

        store = IdentityVectorStore()

        # Add events
        events = [
            {"event_id": "vpn_001", "text": "VPN login from unknown device", "user_id": "user_001"},
            {"event_id": "normal_001", "text": "Normal login from office laptop", "user_id": "user_002"},
            {"event_id": "vpn_002", "text": "VPN connection from new location", "user_id": "user_003"},
        ]
        store.add_events_batch(events, tenant_id="test_tenant")

        # Search for VPN events
        results = store.search(
            query="VPN login",
            tenant_id="test_tenant",
            limit=3,
        )

        assert len(results) > 0
        # VPN events should rank higher
        assert any("vpn" in r.get("text", "").lower() for r in results[:2])

    def test_create_event_text(self):
        """Test event text creation."""
        from src.agents.vector_store import create_event_text

        event = {
            "user_id": "user_001",
            "device_id": "device_unknown",
            "location_country": "RU",
            "mfa_used": False,
            "vpn_detected": True,
            "success": False,
            "risk_score": 0.85,
        }

        text = create_event_text(event)

        assert "user_001" in text
        assert "VPN" in text
        assert "CRITICAL" in text  # CRITICAL risk (score >= 0.8)
        assert "failed" in text.lower()  # Failed attempt


# =============================================================================
# RAG Pipeline Tests
# =============================================================================

class TestRAGPipeline:
    """Tests for the RAG pipeline."""

    def test_rag_pipeline_initialization(self):
        """Test RAG pipeline initializes correctly."""
        from src.agents.rag import RAGPipeline

        pipeline = RAGPipeline()
        assert pipeline is not None
        assert pipeline.vector_store is not None
        assert pipeline.explainer is not None

    def test_ingest_events(self):
        """Test event ingestion."""
        from src.agents.rag import RAGPipeline

        pipeline = RAGPipeline()

        events = [
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
                "device_id": "device_unknown",
                "location_country": "RU",
                "mfa_used": False,
                "vpn_detected": True,
                "success": False,
                "risk_score": 0.9,
            },
        ]

        count = pipeline.ingest_events(events, tenant_id="test")
        assert count == 2

    def test_query_with_event_context(self):
        """Test querying with event context."""
        from src.agents.rag import RAGPipeline

        pipeline = RAGPipeline()

        result = pipeline.query(
            query="Why was this login flagged?",
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

        assert "explanation" in result
        assert result["risk_level"] == "critical"
        assert len(result["factors"]) > 0

    def test_risk_explainer_template(self):
        """Test template-based explanation generation."""
        from src.agents.rag import RiskExplainer

        explainer = RiskExplainer()

        result = explainer.explain(
            event={
                "user_id": "user_001",
                "device_id": "unknown_device",
                "ip": "185.199.1.1",
                "location_country": "RU",
            },
            risk_score=0.75,
            risk_level="high",
            risk_factors=["Unknown device", "VPN detected"],
        )

        assert "HIGH" in result["explanation"]
        assert "step-up" in result["explanation"].lower()  # High risk recommendation


# =============================================================================
# LangGraph Agent Tests
# =============================================================================

class TestLangGraphAgent:
    """Tests for the LangGraph agent."""

    def test_create_risk_agent(self):
        """Test agent creation."""
        from src.agents.agent import create_risk_agent

        agent = create_risk_agent()
        assert agent is not None

    def test_run_risk_agent_normal_login(self):
        """Test agent with normal (low-risk) login."""
        from src.agents.agent import run_risk_agent

        result = run_risk_agent(
            user_id="employee_001",
            device_id="laptop_001",
            ip="10.0.0.1",
            mfa_used=True,
            vpn_detected=False,
        )

        assert result["risk_score"] is not None
        # Normal login should have lower risk (<=0.5)
        assert result["risk_score"] <= 0.5
        assert result["risk_level"] in ["low", "medium"]
        assert result["recommended_action"] is not None

    def test_run_risk_agent_suspicious_login(self):
        """Test agent with suspicious (high-risk) login."""
        from src.agents.agent import run_risk_agent

        result = run_risk_agent(
            user_id="unknown_user",
            device_id="device_unknown_999",
            ip="185.199.1.1",
            mfa_used=False,
            vpn_detected=True,
        )

        assert result["risk_score"] is not None
        # Suspicious login should have higher risk (unknown device + VPN + no MFA)
        assert result["risk_score"] >= 0.4
        assert result["risk_level"] in ["medium", "high", "critical"]
        assert result["recommended_action"] is not None

    def test_identity_tools_schema(self):
        """Test tools have valid schemas."""
        from src.agents.agent import IdentityTools

        tools = IdentityTools()
        schemas = tools.get_tools_schema()

        assert len(schemas) >= 4
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_identity_tools_execution(self):
        """Test tool execution."""
        from src.agents.agent import IdentityTools

        tools = IdentityTools()

        # Test get_risk_score
        result = tools.execute_tool("get_risk_score", {
            "user_id": "user_001",
            "device_id": "laptop_001",
            "ip": "10.0.0.1",
            "mfa_used": True,
        })
        assert "risk_score" in result
        assert "risk_level" in result

        # Test get_user_history
        result = tools.execute_tool("get_user_history", {
            "user_id": "user_001",
            "days": 7,
        })
        assert "total_logins" in result

        # Test quarantine_account
        result = tools.execute_tool("quarantine_account", {
            "user_id": "user_001",
            "reason": "Suspicious activity",
            "duration_hours": 24,
        })
        assert result["status"] == "success"
        assert "mock" in result["note"].lower()  # Should be mock


# =============================================================================
# MCP-like Tools Tests
# =============================================================================

class TestMCPTools:
    """Tests for MCP-like tools."""

    def test_risk_score_tool(self):
        """Test risk score tool."""
        from src.tools.risk_score import RiskScoreInput, execute

        input_data = RiskScoreInput(
            user_id="user_001",
            device_id="laptop_001",
            ip="10.0.0.1",
            mfa_used=True,
            vpn_detected=False,
        )

        result = execute(input_data)

        assert result.user_id == "user_001"
        assert 0 <= result.risk_score <= 1
        assert result.risk_level in ["low", "medium", "high", "critical"]

    def test_risk_score_high_risk_factors(self):
        """Test risk score with high-risk factors."""
        from src.tools.risk_score import RiskScoreInput, execute

        input_data = RiskScoreInput(
            user_id="user_001",
            device_id="device_unknown_123",  # Unknown device
            ip="185.199.1.1",
            location_country="RU",  # High-risk country
            mfa_used=False,  # No MFA
            vpn_detected=True,  # VPN
        )

        result = execute(input_data)

        assert result.risk_score > 0.5
        assert len(result.factors) > 0

    def test_user_history_tool(self):
        """Test user history tool."""
        from src.tools.user_history import UserHistoryInput, execute

        input_data = UserHistoryInput(
            user_id="user_001",
            days=7,
            tenant_id="default",
        )

        result = execute(input_data)

        assert result.user_id == "user_001"
        assert result.period_days == 7
        assert result.summary.total_logins > 0
        assert result.last_login is not None

    def test_quarantine_tool_dry_run(self):
        """Test quarantine tool in dry run mode."""
        from src.tools.quarantine import QuarantineInput, execute

        input_data = QuarantineInput(
            user_id="user_001",
            reason="Suspicious activity detected",
            duration_hours=24,
            notify_user=True,
            notify_admin=True,
        )

        result = execute(input_data, dry_run=True)

        assert result.user_id == "user_001"
        assert result.status == "success"
        assert "DRY RUN" in result.note
        assert "user_email" in result.notifications_sent
        assert "admin_slack" in result.notifications_sent

    def test_quarantine_tool_production(self):
        """Test quarantine tool in production mode."""
        from src.tools.quarantine import QuarantineInput, execute

        input_data = QuarantineInput(
            user_id="user_001",
            reason="Test quarantine",
            duration_hours=1,
        )

        result = execute(input_data, dry_run=False)

        assert result.status == "success"
        assert "DRY RUN" not in result.note

    def test_tool_schemas(self):
        """Test that all tools have valid schemas."""
        from src.tools.risk_score import TOOL_SCHEMA as risk_schema
        from src.tools.user_history import TOOL_SCHEMA as history_schema
        from src.tools.quarantine import TOOL_SCHEMA as quarantine_schema

        for schema in [risk_schema, history_schema, quarantine_schema]:
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert "output_schema" in schema


# =============================================================================
# Privacy/PII Detection Tests
# =============================================================================

class TestPIIDetector:
    """Tests for PII detection."""

    def test_pii_detector_initialization(self):
        """Test PII detector initializes."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        assert detector is not None

    def test_detect_email(self):
        """Test email detection."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.detect("Contact john.doe@example.com for support")

        assert result.has_pii
        assert "[REDACTED]" in result.redacted_text
        assert any(e["type"] == "EMAIL_ADDRESS" for e in result.entities_found)

    def test_detect_phone_number(self):
        """Test phone number detection."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.detect("Call us at 555-123-4567")

        assert result.has_pii
        assert "[REDACTED]" in result.redacted_text

    def test_detect_ip_address(self):
        """Test IP address detection."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.detect("Login from IP 192.168.1.100")

        assert result.has_pii
        assert "[REDACTED]" in result.redacted_text

    def test_detect_ssn(self):
        """Test SSN detection."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.detect("SSN: 123-45-6789")

        assert result.has_pii
        assert "[REDACTED]" in result.redacted_text

    def test_no_pii_in_clean_text(self):
        """Test clean text has no PII detected."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        result = detector.detect("Normal login from San Francisco at 3pm")

        # Should not detect PII in clean text
        assert not result.has_pii or len(result.entities_found) == 0

    def test_redact_convenience_method(self):
        """Test redact convenience method."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        redacted = detector.redact("Email: test@example.com")

        assert "test@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_dict(self):
        """Test dictionary redaction."""
        from src.privacy.pii_detector import PIIDetector

        detector = PIIDetector()
        data = {
            "user_id": "user_001",
            "email": "john@example.com",
            "phone": "555-123-4567",
            "notes": "Normal login",
        }

        redacted = detector.redact_dict(data)

        assert "user_001" in redacted["user_id"]  # IDs should stay
        assert "[REDACTED]" in redacted["email"]
        assert "[REDACTED]" in redacted["phone"]

    def test_privacy_middleware(self):
        """Test privacy middleware."""
        from src.privacy.pii_detector import PrivacyMiddleware

        middleware = PrivacyMiddleware()

        result = middleware.process_for_llm(
            text="User john.doe@example.com logged in from 192.168.1.1",
            context={"user_email": "john.doe@example.com"},
        )

        assert result["pii_detected"]
        assert result["entity_count"] >= 2
        assert "[REDACTED]" in result["text"]
        assert "context" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase3Integration:
    """Integration tests for Phase 3 components."""

    def test_full_risk_analysis_flow(self):
        """Test complete flow from event to explanation."""
        from src.agents.agent import run_risk_agent
        from src.agents.rag import RAGPipeline
        from src.privacy.pii_detector import PIIDetector

        # 1. Analyze event with agent
        agent_result = run_risk_agent(
            user_id="suspicious_user",
            device_id="device_unknown_123",
            ip="185.199.1.1",
            mfa_used=False,
            vpn_detected=True,
        )

        # Agent should detect elevated risk for suspicious login
        assert agent_result["risk_level"] in ["medium", "high", "critical"]

        # 2. Get explanation via RAG
        pipeline = RAGPipeline()
        explanation = pipeline.query(
            query="Why was this login flagged?",
            event={
                "user_id": "suspicious_user",
                "device_id": "device_unknown_123",
                "vpn_detected": True,
            },
            risk_score=agent_result["risk_score"],
            risk_level=agent_result["risk_level"],
            risk_factors=["Unknown device", "VPN detected"],
        )

        assert "explanation" in explanation
        assert agent_result["risk_level"].upper() in explanation["explanation"]

        # 3. Verify PII protection
        detector = PIIDetector()
        result = detector.detect(
            f"User john.doe@example.com had risk score {agent_result['risk_score']}"
        )
        assert result.has_pii
        assert "john.doe@example.com" not in result.redacted_text

    def test_multi_tenant_isolation(self):
        """Test that vector store respects tenant isolation."""
        from src.agents.vector_store import IdentityVectorStore

        store = IdentityVectorStore()

        # Add events for different tenants
        store.add_event(
            {"event_id": "tenant_a_001", "text": "VPN login tenant A", "user_id": "user_a"},
            tenant_id="tenant_a",
        )
        store.add_event(
            {"event_id": "tenant_b_001", "text": "VPN login tenant B", "user_id": "user_b"},
            tenant_id="tenant_b",
        )

        # Search with tenant filter
        results_a = store.search("VPN login", tenant_id="tenant_a", limit=10)
        results_b = store.search("VPN login", tenant_id="tenant_b", limit=10)

        # Each should only see their own events
        for r in results_a:
            if "tenant_id" in r:
                assert r["tenant_id"] == "tenant_a"

        for r in results_b:
            if "tenant_id" in r:
                assert r["tenant_id"] == "tenant_b"
