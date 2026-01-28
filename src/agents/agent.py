"""
LangGraph Agent for Identity Risk Analysis

Autonomous agent that can investigate security events using tools.
"""

import logging
from typing import Annotated, List, Optional, Dict, Any, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict):
    """State for the identity risk agent."""
    messages: Annotated[list, add_messages]
    user_id: Optional[str]
    tenant_id: str
    event_id: Optional[str]
    risk_score: Optional[float]
    risk_level: Optional[str]
    risk_factors: List[str]
    investigation_notes: List[str]
    recommended_action: Optional[str]


# =============================================================================
# Tools (MCP-like)
# =============================================================================

class IdentityTools:
    """
    MCP-like tools for the identity risk agent.

    Each tool has a schema and implementation that can be called by the agent.
    """

    def __init__(self, risk_scorer=None, vector_store=None, feature_store=None):
        self.risk_scorer = risk_scorer
        self.vector_store = vector_store
        self.feature_store = feature_store

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return tool schemas for the LLM."""
        return [
            {
                "name": "get_risk_score",
                "description": "Calculate the risk score for a login event",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "device_id": {"type": "string", "description": "Device identifier"},
                        "ip": {"type": "string", "description": "IP address"},
                        "location_country": {"type": "string", "description": "Country code"},
                        "mfa_used": {"type": "boolean", "description": "Whether MFA was used"},
                        "vpn_detected": {"type": "boolean", "description": "Whether VPN was detected"},
                    },
                    "required": ["user_id", "device_id", "ip"],
                },
            },
            {
                "name": "get_user_history",
                "description": "Get recent login history for a user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User identifier"},
                        "days": {"type": "integer", "description": "Number of days to look back"},
                    },
                    "required": ["user_id"],
                },
            },
            {
                "name": "search_similar_events",
                "description": "Search for similar login events in history",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "quarantine_account",
                "description": "Temporarily quarantine a user account (mock action)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "User to quarantine"},
                        "reason": {"type": "string", "description": "Reason for quarantine"},
                        "duration_hours": {"type": "integer", "description": "Duration in hours"},
                    },
                    "required": ["user_id", "reason"],
                },
            },
            {
                "name": "create_security_alert",
                "description": "Create a security alert for investigation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                        "title": {"type": "string", "description": "Alert title"},
                        "description": {"type": "string", "description": "Alert details"},
                        "user_id": {"type": "string", "description": "Related user"},
                    },
                    "required": ["severity", "title"],
                },
            },
        ]

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return results."""
        tool_map = {
            "get_risk_score": self._get_risk_score,
            "get_user_history": self._get_user_history,
            "search_similar_events": self._search_similar_events,
            "quarantine_account": self._quarantine_account,
            "create_security_alert": self._create_security_alert,
        }

        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool_map[tool_name](**args)
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return {"error": str(e)}

    def _get_risk_score(
        self,
        user_id: str,
        device_id: str,
        ip: str,
        location_country: str = "US",
        mfa_used: bool = False,
        vpn_detected: bool = False,
    ) -> Dict[str, Any]:
        """Calculate risk score for a login."""
        if self.risk_scorer:
            # Use actual risk scorer
            features = {
                "failed_logins_24h": 0,
                "login_count_7d": 0,
                "device_age_days": 0 if "unknown" in device_id.lower() else 30,
                "is_new_device": 1 if "unknown" in device_id.lower() else 0,
                "ip_reputation_score": 0.5 if vpn_detected else 0.1,
                "hour_of_day": datetime.now().hour,
                "is_unusual_hour": 0,
                "location_changed": 0,
                "mfa_used": 1 if mfa_used else 0,
                "vpn_detected": 1 if vpn_detected else 0,
                "success": 1,
            }
            result = self.risk_scorer.score(features)
        else:
            # Mock scoring
            score = 0.1
            if "unknown" in device_id.lower():
                score += 0.3
            if vpn_detected:
                score += 0.2
            if not mfa_used:
                score += 0.1
            if location_country in ["RU", "CN", "KP", "IR"]:
                score += 0.3

            score = min(score, 1.0)
            level = "low" if score < 0.3 else "medium" if score < 0.6 else "high" if score < 0.8 else "critical"
            result = {"risk_score": score, "risk_level": level}

        return {
            "user_id": user_id,
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "timestamp": datetime.now().isoformat(),
        }

    def _get_user_history(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get user login history (mock)."""
        # Mock history - in production would query actual data
        return {
            "user_id": user_id,
            "period_days": days,
            "total_logins": 15,
            "failed_logins": 2,
            "unique_devices": 2,
            "unique_locations": 1,
            "last_login": "2026-01-27T14:30:00Z",
            "typical_hours": "9:00-18:00 UTC",
        }

    def _search_similar_events(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search for similar events."""
        if self.vector_store:
            results = self.vector_store.search(query=query, limit=limit)
            return {"query": query, "results": results, "count": len(results)}
        else:
            # Mock results
            return {
                "query": query,
                "results": [
                    {"event_id": "evt_001", "similarity": 0.85, "summary": "Similar VPN login from unknown device"},
                    {"event_id": "evt_002", "similarity": 0.72, "summary": "Login attempt from same IP range"},
                ],
                "count": 2,
            }

    def _quarantine_account(
        self,
        user_id: str,
        reason: str,
        duration_hours: int = 24,
    ) -> Dict[str, Any]:
        """Quarantine a user account (mock)."""
        logger.warning(f"MOCK: Quarantining user {user_id} for {duration_hours}h: {reason}")
        return {
            "action": "quarantine",
            "user_id": user_id,
            "reason": reason,
            "duration_hours": duration_hours,
            "status": "success",
            "note": "This is a mock action - no actual quarantine applied",
        }

    def _create_security_alert(
        self,
        severity: str,
        title: str,
        description: str = "",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a security alert (mock)."""
        alert_id = f"alert_{hash(title) % 10000:04d}"
        logger.info(f"MOCK: Created {severity} alert: {title}")
        return {
            "alert_id": alert_id,
            "severity": severity,
            "title": title,
            "description": description,
            "user_id": user_id,
            "status": "created",
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Agent Graph
# =============================================================================

def create_risk_agent(
    tools: Optional[IdentityTools] = None,
    llm=None,
) -> StateGraph:
    """
    Create a LangGraph agent for identity risk analysis.

    Args:
        tools: IdentityTools instance
        llm: LangChain LLM (optional, uses rule-based logic if None)

    Returns:
        Compiled StateGraph
    """
    tools = tools or IdentityTools()

    def analyze_event(state: AgentState) -> AgentState:
        """Analyze the current event and determine risk."""
        messages = state.get("messages", [])
        user_id = state.get("user_id")
        event_id = state.get("event_id")

        # Get risk score if we have user info
        if user_id:
            result = tools.execute_tool("get_risk_score", {
                "user_id": user_id,
                "device_id": state.get("device_id", "unknown"),
                "ip": state.get("ip", "0.0.0.0"),
                "mfa_used": state.get("mfa_used", False),
                "vpn_detected": state.get("vpn_detected", False),
            })
            state["risk_score"] = result.get("risk_score", 0)
            state["risk_level"] = result.get("risk_level", "unknown")

        return state

    def investigate(state: AgentState) -> AgentState:
        """Investigate high-risk events."""
        risk_level = state.get("risk_level", "low")
        user_id = state.get("user_id")
        notes = state.get("investigation_notes", [])

        if risk_level in ["high", "critical"]:
            # Get user history
            if user_id:
                history = tools.execute_tool("get_user_history", {"user_id": user_id})
                notes.append(f"User history: {history['total_logins']} logins, {history['failed_logins']} failed")

            # Search similar events
            similar = tools.execute_tool("search_similar_events", {
                "query": f"suspicious login {state.get('risk_level')} risk",
                "limit": 3,
            })
            notes.append(f"Found {similar['count']} similar events")

        state["investigation_notes"] = notes
        return state

    def decide_action(state: AgentState) -> AgentState:
        """Decide on recommended action."""
        risk_level = state.get("risk_level", "low")
        risk_score = state.get("risk_score", 0)

        if risk_level == "critical":
            action = "Block login and create critical security alert"
            # Mock: Create alert
            tools.execute_tool("create_security_alert", {
                "severity": "critical",
                "title": f"Critical risk login detected for {state.get('user_id')}",
                "user_id": state.get("user_id"),
            })
        elif risk_level == "high":
            action = "Require step-up authentication and monitor session"
        elif risk_level == "medium":
            action = "Allow with enhanced monitoring"
        else:
            action = "Allow - normal risk profile"

        state["recommended_action"] = action

        # Add response message
        response = f"""**Risk Analysis Complete**

Risk Score: {risk_score:.2f}
Risk Level: {risk_level.upper()}

Investigation Notes:
{chr(10).join('- ' + n for n in state.get('investigation_notes', ['No additional investigation needed']))}

Recommended Action: {action}
"""
        state["messages"] = state.get("messages", []) + [AIMessage(content=response)]

        return state

    def should_investigate(state: AgentState) -> str:
        """Determine if investigation is needed."""
        risk_level = state.get("risk_level", "low")
        if risk_level in ["high", "critical"]:
            return "investigate"
        return "decide"

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("analyze", analyze_event)
    workflow.add_node("investigate", investigate)
    workflow.add_node("decide", decide_action)

    # Add edges
    workflow.set_entry_point("analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_investigate,
        {
            "investigate": "investigate",
            "decide": "decide",
        }
    )
    workflow.add_edge("investigate", "decide")
    workflow.add_edge("decide", END)

    return workflow.compile()


# Convenience function for simple invocation
def run_risk_agent(
    user_id: str,
    device_id: str = "unknown",
    ip: str = "0.0.0.0",
    tenant_id: str = "default",
    mfa_used: bool = False,
    vpn_detected: bool = False,
    tools: Optional[IdentityTools] = None,
) -> Dict[str, Any]:
    """
    Run the risk analysis agent.

    Args:
        user_id: User identifier
        device_id: Device identifier
        ip: IP address
        tenant_id: Tenant identifier
        mfa_used: Whether MFA was used
        vpn_detected: Whether VPN was detected
        tools: Optional tools instance

    Returns:
        Agent result with risk assessment
    """
    agent = create_risk_agent(tools=tools)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=f"Analyze login for user {user_id}")],
        "user_id": user_id,
        "tenant_id": tenant_id,
        "event_id": None,
        "device_id": device_id,
        "ip": ip,
        "mfa_used": mfa_used,
        "vpn_detected": vpn_detected,
        "risk_score": None,
        "risk_level": None,
        "risk_factors": [],
        "investigation_notes": [],
        "recommended_action": None,
    }

    result = agent.invoke(initial_state)

    return {
        "risk_score": result.get("risk_score"),
        "risk_level": result.get("risk_level"),
        "recommended_action": result.get("recommended_action"),
        "investigation_notes": result.get("investigation_notes"),
        "messages": [m.content for m in result.get("messages", []) if hasattr(m, "content")],
    }


if __name__ == "__main__":
    print("Testing LangGraph agent...")

    # Test normal login
    print("\n=== Normal Login ===")
    result = run_risk_agent(
        user_id="employee_001",
        device_id="laptop_001",
        mfa_used=True,
        vpn_detected=False,
    )
    print(f"Risk: {result['risk_level']} ({result['risk_score']:.2f})")
    print(f"Action: {result['recommended_action']}")

    # Test suspicious login
    print("\n=== Suspicious Login ===")
    result = run_risk_agent(
        user_id="unknown_user",
        device_id="device_unknown_999",
        ip="185.199.1.1",
        mfa_used=False,
        vpn_detected=True,
    )
    print(f"Risk: {result['risk_level']} ({result['risk_score']:.2f})")
    print(f"Action: {result['recommended_action']}")
    print(f"Notes: {result['investigation_notes']}")
