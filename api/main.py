"""
Identity Risk Agent - FastAPI Application

Main API endpoints for risk scoring and agent interactions.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.risk_model import RiskScorer, FEATURE_COLUMNS
from src.agents.rag import RAGPipeline
from src.privacy.pii_detector import PIIDetector, PrivacyMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global references
_scorer: Optional[RiskScorer] = None
_rag_pipeline: Optional[RAGPipeline] = None
_privacy: Optional[PrivacyMiddleware] = None


def get_scorer() -> RiskScorer:
    """Get the loaded risk scorer model."""
    if _scorer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is starting up or model is missing."
        )
    return _scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _scorer, _rag_pipeline, _privacy

    # Load risk model
    model_path = Path("models/risk_model.pkl")
    if model_path.exists():
        logger.info(f"Loading model from {model_path}")
        _scorer = RiskScorer.load(str(model_path))
        logger.info("Model loaded successfully")
    else:
        logger.warning(f"Model not found at {model_path}. Scoring will return mock results.")

    # Initialize RAG pipeline
    try:
        _rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized")
    except Exception as e:
        logger.warning(f"RAG pipeline initialization failed: {e}. Explain endpoint will use fallback.")

    # Initialize privacy middleware
    try:
        _privacy = PrivacyMiddleware(PIIDetector())
        logger.info("Privacy middleware initialized")
    except Exception as e:
        logger.warning(f"Privacy middleware initialization failed: {e}")

    yield

    # Cleanup
    _scorer = None
    _rag_pipeline = None
    _privacy = None
    logger.info("Resources unloaded")


app = FastAPI(
    title="Identity Risk Agent API",
    description="ML-powered identity risk scoring and agentic explanations",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware for UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class LoginEvent(BaseModel):
    """Input schema for login event scoring."""
    user_id: str = Field(..., description="Unique user identifier")
    tenant_id: str = Field(default="default", description="Tenant identifier for multi-tenant isolation")
    ip: str = Field(..., description="IP address of the login attempt")
    device_id: str = Field(..., description="Device identifier")
    location_country: str = Field(default="US", description="Country code of login location")
    location_city: Optional[str] = Field(default=None, description="City of login location")
    timestamp: Optional[datetime] = Field(default=None, description="Login timestamp")
    mfa_used: bool = Field(default=False, description="Whether MFA was used")
    vpn_detected: bool = Field(default=False, description="Whether VPN was detected")
    success: bool = Field(default=True, description="Whether login was successful")

    # Optional pre-computed features (for testing/advanced use)
    failed_logins_24h: Optional[int] = Field(default=0, description="Failed logins in past 24h")
    login_count_7d: Optional[int] = Field(default=0, description="Login count in past 7 days")
    device_age_days: Optional[float] = Field(default=0, description="Days since device first seen")


class RiskScore(BaseModel):
    """Output schema for risk scoring."""
    user_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Risk score from 0 to 1")
    risk_level: str = Field(..., description="Risk level: low, medium, high, critical")
    factors: list[str] = Field(default_factory=list, description="Contributing risk factors")
    timestamp: datetime
    model_version: str = Field(default="0.1.0")


class ExplainRequest(BaseModel):
    """Input for agent explanation."""
    query: str = Field(..., description="Question about the risk decision")
    login_id: Optional[str] = Field(default=None, description="Optional login event ID")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    # Optional event context for explanation
    event: Optional[LoginEvent] = Field(default=None, description="Event to explain")
    risk_score: Optional[float] = Field(default=None, ge=0, le=1)
    risk_level: Optional[str] = Field(default=None)
    risk_factors: Optional[list[str]] = Field(default=None)


class ExplainResponse(BaseModel):
    """Output from agent explanation."""
    explanation: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


# Risk country mapping for IP reputation
RISK_COUNTRIES = {"RU", "CN", "KP", "IR", "NG"}


def compute_features(event: LoginEvent) -> dict:
    """
    Compute features from a login event for model scoring.

    In production, this would query the feature store.
    For MVP, we compute features directly from the event.
    """
    # Compute IP reputation score (mock)
    ip_reputation = 0.0
    if event.vpn_detected:
        ip_reputation += 0.3
    if event.location_country in RISK_COUNTRIES:
        ip_reputation += 0.5
    if "unknown" in event.device_id.lower():
        ip_reputation += 0.2
    ip_reputation = min(ip_reputation, 1.0)

    # Compute hour-based features
    ts = event.timestamp or datetime.now()
    hour = ts.hour
    is_unusual_hour = 1 if (hour < 6 or hour > 22) else 0

    # Check for new device (simplified: unknown devices are new)
    is_new_device = 1 if event.device_age_days == 0 or "unknown" in event.device_id.lower() else 0

    # Build feature dict
    features = {
        "failed_logins_24h": event.failed_logins_24h or 0,
        "login_count_7d": event.login_count_7d or 0,
        "device_age_days": event.device_age_days or 0,
        "is_new_device": is_new_device,
        "ip_reputation_score": ip_reputation,
        "hour_of_day": hour,
        "is_unusual_hour": is_unusual_hour,
        "location_changed": 0,  # Would need history to compute
        "mfa_used": 1 if event.mfa_used else 0,
        "vpn_detected": 1 if event.vpn_detected else 0,
        "success": 1 if event.success else 0,
    }

    return features


def get_risk_factors(features: dict, risk_score: float) -> list[str]:
    """Identify contributing risk factors based on features."""
    factors = []

    if features["is_new_device"]:
        factors.append("New or unknown device")

    if features["ip_reputation_score"] > 0.3:
        factors.append("Suspicious IP reputation")

    if features["is_unusual_hour"]:
        factors.append("Login at unusual hour")

    if features["vpn_detected"]:
        factors.append("VPN detected")

    if not features["mfa_used"]:
        factors.append("MFA not used")

    if not features["success"]:
        factors.append("Failed login attempt")

    if features["failed_logins_24h"] > 2:
        factors.append(f"{features['failed_logins_24h']} failed logins in 24h")

    return factors


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = _scorer is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "version": "0.1.0",
        "model_loaded": model_loaded,
    }


@app.post("/score", response_model=RiskScore)
async def score_login(event: LoginEvent):
    """
    Score a login event for risk.

    Returns a risk score (0-1) with contributing factors.
    """
    logger.info(f"Scoring login for user={event.user_id}, tenant={event.tenant_id}")

    # Compute features
    features = compute_features(event)

    # Get risk score
    if _scorer is not None:
        result = _scorer.score(features)
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]
    else:
        # Fallback: simple heuristic if model not loaded
        risk_score = features["ip_reputation_score"] * 0.5 + features["is_new_device"] * 0.3
        risk_score = min(risk_score, 1.0)
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "medium"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"

    # Get risk factors
    factors = get_risk_factors(features, risk_score)

    return RiskScore(
        user_id=event.user_id,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        factors=factors,
        timestamp=event.timestamp or datetime.now(),
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain_risk(request: ExplainRequest):
    """
    Get an AI-powered explanation for a risk decision.

    Uses RAG to retrieve relevant context and generate explanations.
    """
    logger.info(f"Explain request: query={request.query[:50]}...")

    # Apply privacy controls to query
    processed_query = request.query
    if _privacy:
        result = _privacy.process_for_llm(request.query)
        processed_query = result["text"]
        if result["pii_detected"]:
            logger.info(f"PII redacted from query: {result['entity_count']} entities")

    # Use RAG pipeline if available
    if _rag_pipeline:
        try:
            # Build event dict if provided
            event_dict = None
            if request.event:
                event_dict = {
                    "user_id": request.event.user_id,
                    "device_id": request.event.device_id,
                    "ip": request.event.ip,
                    "location_country": request.event.location_country,
                    "location_city": request.event.location_city,
                    "mfa_used": request.event.mfa_used,
                    "vpn_detected": request.event.vpn_detected,
                    "success": request.event.success,
                    "timestamp": str(request.event.timestamp) if request.event.timestamp else None,
                }

            result = _rag_pipeline.query(
                query=processed_query,
                event=event_dict,
                risk_score=request.risk_score or 0.0,
                risk_level=request.risk_level or "low",
                risk_factors=request.risk_factors or [],
                tenant_id=request.tenant_id,
            )

            return ExplainResponse(
                explanation=result["explanation"],
                sources=result.get("sources", []),
                confidence=0.8 if event_dict else 0.5,
            )
        except Exception as e:
            logger.error(f"RAG pipeline error: {e}")
            # Fall through to fallback response

    # Fallback response
    explanation = (
        f"Query: {processed_query}\n\n"
        "Unable to generate detailed explanation. RAG pipeline not available."
    )

    return ExplainResponse(
        explanation=explanation,
        sources=[],
        confidence=0.0,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Identity Risk Agent API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /score": "Score a login event for risk",
            "POST /explain": "Get AI explanation for risk decision (Phase 3)",
        },
    }


@app.get("/features")
async def list_features():
    """List the features used by the model."""
    return {
        "features": FEATURE_COLUMNS,
        "description": {
            "failed_logins_24h": "Number of failed logins in past 24 hours",
            "login_count_7d": "Total logins in past 7 days",
            "device_age_days": "Days since device first seen",
            "is_new_device": "Whether device is new (0 or 1)",
            "ip_reputation_score": "IP risk score (0-1)",
            "hour_of_day": "Hour of login (0-23)",
            "is_unusual_hour": "Login outside 6am-10pm (0 or 1)",
            "location_changed": "Location differs from previous (0 or 1)",
            "mfa_used": "MFA was used (0 or 1)",
            "vpn_detected": "VPN detected (0 or 1)",
            "success": "Login successful (0 or 1)",
        },
    }
