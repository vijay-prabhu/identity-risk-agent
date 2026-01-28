"""
Quarantine Tool

MCP-like tool for quarantining user accounts (mock action).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QuarantineInput(BaseModel):
    """Input schema for quarantine tool."""
    user_id: str = Field(..., description="User identifier to quarantine")
    reason: str = Field(..., description="Reason for quarantine")
    duration_hours: int = Field(default=24, ge=1, le=720, description="Duration in hours")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    notify_user: bool = Field(default=True, description="Whether to notify the user")
    notify_admin: bool = Field(default=True, description="Whether to notify admins")


class QuarantineOutput(BaseModel):
    """Output schema for quarantine tool."""
    action_id: str
    user_id: str
    status: str  # "success", "failed", "pending"
    reason: str
    started_at: datetime
    expires_at: datetime
    notifications_sent: list[str]
    note: str


TOOL_SCHEMA = {
    "name": "quarantine_account",
    "description": "Temporarily quarantine a user account to prevent further access. This is a security action that should be used for high-risk situations.",
    "input_schema": QuarantineInput.model_json_schema(),
    "output_schema": QuarantineOutput.model_json_schema(),
}


def execute(
    input_data: QuarantineInput,
    dry_run: bool = True,
) -> QuarantineOutput:
    """
    Execute the quarantine tool.

    Args:
        input_data: Validated input
        dry_run: If True, don't actually quarantine (mock mode)

    Returns:
        Quarantine action result
    """
    action_id = f"quar_{hash(input_data.user_id + str(datetime.now())) % 100000:05d}"
    started_at = datetime.now()
    expires_at = started_at + timedelta(hours=input_data.duration_hours)

    notifications = []
    if input_data.notify_user:
        notifications.append("user_email")
    if input_data.notify_admin:
        notifications.append("admin_slack")
        notifications.append("security_team_email")

    if dry_run:
        logger.warning(
            f"MOCK QUARANTINE: User {input_data.user_id} would be quarantined for "
            f"{input_data.duration_hours}h. Reason: {input_data.reason}"
        )
        note = "DRY RUN - No actual quarantine applied. In production, this would disable the user's access."
        status = "success"
    else:
        # In production, this would:
        # 1. Call identity provider API to disable user
        # 2. Revoke active sessions
        # 3. Send notifications
        # 4. Log to audit trail
        logger.critical(
            f"QUARANTINE: User {input_data.user_id} quarantined for "
            f"{input_data.duration_hours}h. Reason: {input_data.reason}"
        )
        note = "Account quarantined successfully. User will be unable to login until expiry or manual release."
        status = "success"

    return QuarantineOutput(
        action_id=action_id,
        user_id=input_data.user_id,
        status=status,
        reason=input_data.reason,
        started_at=started_at,
        expires_at=expires_at,
        notifications_sent=notifications,
        note=note,
    )
