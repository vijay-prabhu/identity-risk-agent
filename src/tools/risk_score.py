"""
Risk Score Tool

MCP-like tool for calculating identity risk scores.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class RiskScoreInput(BaseModel):
    """Input schema for risk scoring tool."""
    user_id: str = Field(..., description="User identifier")
    device_id: str = Field(..., description="Device identifier")
    ip: str = Field(..., description="IP address of the login attempt")
    location_country: str = Field(default="US", description="Country code")
    mfa_used: bool = Field(default=False, description="Whether MFA was used")
    vpn_detected: bool = Field(default=False, description="Whether VPN was detected")
    success: bool = Field(default=True, description="Whether login was successful")


class RiskScoreOutput(BaseModel):
    """Output schema for risk scoring tool."""
    user_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk score 0-1")
    risk_level: str = Field(..., description="Risk level: low/medium/high/critical")
    factors: list[str] = Field(default_factory=list, description="Contributing factors")
    timestamp: datetime


TOOL_SCHEMA = {
    "name": "risk_score",
    "description": "Calculate the identity risk score for a login attempt. Returns a score from 0 (safe) to 1 (dangerous) with contributing factors.",
    "input_schema": RiskScoreInput.model_json_schema(),
    "output_schema": RiskScoreOutput.model_json_schema(),
}


def execute(
    input_data: RiskScoreInput,
    scorer=None,
) -> RiskScoreOutput:
    """
    Execute the risk scoring tool.

    Args:
        input_data: Validated input
        scorer: Optional RiskScorer instance

    Returns:
        Risk score output
    """
    factors = []

    # Calculate score based on inputs
    if scorer:
        # Use ML model
        features = {
            "failed_logins_24h": 0,
            "login_count_7d": 0,
            "device_age_days": 0 if "unknown" in input_data.device_id.lower() else 30,
            "is_new_device": 1 if "unknown" in input_data.device_id.lower() else 0,
            "ip_reputation_score": 0.5 if input_data.vpn_detected else 0.1,
            "hour_of_day": datetime.now().hour,
            "is_unusual_hour": 0,
            "location_changed": 0,
            "mfa_used": 1 if input_data.mfa_used else 0,
            "vpn_detected": 1 if input_data.vpn_detected else 0,
            "success": 1 if input_data.success else 0,
        }
        result = scorer.score(features)
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]
    else:
        # Rule-based fallback
        risk_score = 0.0

        if "unknown" in input_data.device_id.lower():
            risk_score += 0.3
            factors.append("Unknown or new device")

        if input_data.vpn_detected:
            risk_score += 0.2
            factors.append("VPN detected")

        if not input_data.mfa_used:
            risk_score += 0.15
            factors.append("MFA not used")

        if input_data.location_country in ["RU", "CN", "KP", "IR", "NG"]:
            risk_score += 0.3
            factors.append(f"High-risk location: {input_data.location_country}")

        if not input_data.success:
            risk_score += 0.1
            factors.append("Failed login attempt")

        risk_score = min(risk_score, 1.0)

        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "medium"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"

    return RiskScoreOutput(
        user_id=input_data.user_id,
        risk_score=risk_score,
        risk_level=risk_level,
        factors=factors,
        timestamp=datetime.now(),
    )
