"""
User History Tool

MCP-like tool for retrieving user login history.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class UserHistoryInput(BaseModel):
    """Input schema for user history tool."""
    user_id: str = Field(..., description="User identifier to lookup")
    days: int = Field(default=7, ge=1, le=90, description="Number of days to look back")
    tenant_id: str = Field(default="default", description="Tenant identifier")


class LoginSummary(BaseModel):
    """Summary of login activity."""
    total_logins: int
    successful_logins: int
    failed_logins: int
    unique_devices: int
    unique_locations: int
    unique_ips: int


class UserHistoryOutput(BaseModel):
    """Output schema for user history tool."""
    user_id: str
    tenant_id: str
    period_days: int
    summary: LoginSummary
    last_login: Optional[datetime]
    typical_login_hours: str
    risk_indicators: List[str]


TOOL_SCHEMA = {
    "name": "user_history",
    "description": "Retrieve login history and behavioral patterns for a user. Useful for understanding normal vs anomalous behavior.",
    "input_schema": UserHistoryInput.model_json_schema(),
    "output_schema": UserHistoryOutput.model_json_schema(),
}


def execute(
    input_data: UserHistoryInput,
    data_source=None,
) -> UserHistoryOutput:
    """
    Execute the user history tool.

    Args:
        input_data: Validated input
        data_source: Optional data source for actual history

    Returns:
        User history output
    """
    # In production, this would query the feature store or database
    # For now, return mock data

    risk_indicators = []

    # Mock history data
    summary = LoginSummary(
        total_logins=45,
        successful_logins=43,
        failed_logins=2,
        unique_devices=2,
        unique_locations=1,
        unique_ips=3,
    )

    # Check for risk indicators
    if summary.failed_logins > 5:
        risk_indicators.append(f"High number of failed logins: {summary.failed_logins}")

    if summary.unique_devices > 5:
        risk_indicators.append(f"Many unique devices: {summary.unique_devices}")

    if summary.unique_locations > 3:
        risk_indicators.append(f"Multiple locations: {summary.unique_locations}")

    return UserHistoryOutput(
        user_id=input_data.user_id,
        tenant_id=input_data.tenant_id,
        period_days=input_data.days,
        summary=summary,
        last_login=datetime.now() - timedelta(hours=2),
        typical_login_hours="09:00-18:00 UTC",
        risk_indicators=risk_indicators,
    )
