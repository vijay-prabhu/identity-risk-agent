"""
Identity Risk Agent - Streamlit Dashboard

Interactive UI for risk scoring and agent explanations.
"""

import httpx
import streamlit as st
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Identity Risk Agent",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-critical { color: #dc3545; font-weight: bold; }
    .factor-tag {
        background-color: #f0f2f6;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_color(level: str) -> str:
    """Get color class for risk level."""
    colors = {
        "low": "risk-low",
        "medium": "risk-medium",
        "high": "risk-high",
        "critical": "risk-critical",
    }
    return colors.get(level, "")


def check_api_health(api_url: str) -> dict:
    """Check if API is healthy."""
    try:
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}


# Header
st.title("🔐 Identity Risk Agent")
st.markdown("**ML-powered identity risk scoring with explainable decisions**")

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input("API URL", value=API_URL)

# API Status
health = check_api_health(api_url)
if health.get("status") == "healthy":
    st.sidebar.success("✓ API Connected")
    st.sidebar.caption(f"Model loaded: {health.get('model_loaded', False)}")
elif health.get("status") == "degraded":
    st.sidebar.warning("⚠ API Degraded (model not loaded)")
else:
    st.sidebar.error(f"✗ API Error: {health.get('error', 'Unknown')}")

st.sidebar.markdown("---")
st.sidebar.header("Quick Presets")

preset = st.sidebar.selectbox(
    "Load preset",
    ["Custom", "Normal Employee", "Suspicious Login", "VPN User", "New Device"]
)

# Preset configurations
presets = {
    "Normal Employee": {
        "user_id": "employee_001",
        "ip": "10.0.0.50",
        "device_id": "corporate_laptop_001",
        "location_country": "US",
        "mfa_used": True,
        "vpn_detected": False,
        "success": True,
        "device_age_days": 90.0,
    },
    "Suspicious Login": {
        "user_id": "unknown_user",
        "ip": "185.199.1.100",
        "device_id": "device_unknown_9999",
        "location_country": "RU",
        "mfa_used": False,
        "vpn_detected": True,
        "success": False,
        "device_age_days": 0.0,
    },
    "VPN User": {
        "user_id": "remote_worker_001",
        "ip": "104.238.50.100",
        "device_id": "personal_laptop_001",
        "location_country": "US",
        "mfa_used": True,
        "vpn_detected": True,
        "success": True,
        "device_age_days": 30.0,
    },
    "New Device": {
        "user_id": "employee_002",
        "ip": "192.168.1.100",
        "device_id": "new_phone_001",
        "location_country": "US",
        "mfa_used": True,
        "vpn_detected": False,
        "success": True,
        "device_age_days": 0.0,
    },
}

# Main content
tab1, tab2, tab3 = st.tabs(["🎯 Score Login", "💬 Explain Risk", "📊 Dashboard"])

with tab1:
    st.header("Score Login Event")
    st.markdown("Enter login details to get a risk assessment.")

    col1, col2 = st.columns(2)

    # Get preset values or defaults
    p = presets.get(preset, {})

    with col1:
        st.subheader("User & Device")
        user_id = st.text_input("User ID", value=p.get("user_id", "user_001"))
        device_id = st.text_input("Device ID", value=p.get("device_id", "device_001"))
        device_age = st.number_input(
            "Device Age (days)",
            min_value=0.0,
            value=p.get("device_age_days", 0.0),
            help="Days since this device was first seen for this user"
        )

    with col2:
        st.subheader("Network & Location")
        ip_address = st.text_input("IP Address", value=p.get("ip", "192.168.1.1"))
        location = st.selectbox(
            "Country",
            ["US", "CA", "UK", "DE", "FR", "JP", "AU", "RU", "CN", "IR", "Other"],
            index=["US", "CA", "UK", "DE", "FR", "JP", "AU", "RU", "CN", "IR", "Other"].index(p.get("location_country", "US"))
        )

    st.subheader("Security Factors")
    col3, col4, col5 = st.columns(3)

    with col3:
        mfa_used = st.checkbox("MFA Used", value=p.get("mfa_used", False))
    with col4:
        vpn_detected = st.checkbox("VPN Detected", value=p.get("vpn_detected", False))
    with col5:
        success = st.checkbox("Login Successful", value=p.get("success", True))

    st.markdown("---")

    if st.button("🔍 Score Login", type="primary", use_container_width=True):
        payload = {
            "user_id": user_id,
            "ip": ip_address,
            "device_id": device_id,
            "location_country": location,
            "mfa_used": mfa_used,
            "vpn_detected": vpn_detected,
            "success": success,
            "device_age_days": device_age,
        }

        try:
            with st.spinner("Analyzing login..."):
                response = httpx.post(f"{api_url}/score", json=payload, timeout=10.0)
                result = response.json()

            # Display results
            st.markdown("---")
            st.subheader("Risk Assessment")

            # Main metrics
            col1, col2, col3 = st.columns(3)

            risk_score = result.get("risk_score", 0)
            risk_level = result.get("risk_level", "unknown")

            with col1:
                st.metric(
                    "Risk Score",
                    f"{risk_score:.2f}",
                    delta=None,
                )
                # Progress bar for visual
                st.progress(risk_score)

            with col2:
                color_class = get_risk_color(risk_level)
                st.markdown(f"**Risk Level**")
                st.markdown(f"<span class='{color_class}'>{risk_level.upper()}</span>", unsafe_allow_html=True)

            with col3:
                st.metric("Risk Factors", len(result.get("factors", [])))

            # Risk factors
            factors = result.get("factors", [])
            if factors:
                st.subheader("Contributing Factors")
                for factor in factors:
                    st.markdown(f"- ⚠️ {factor}")
            else:
                st.success("✓ No risk factors detected")

            # Recommendation
            st.subheader("Recommendation")
            if risk_level == "critical":
                st.error("🚨 **Block login** and alert security team immediately.")
            elif risk_level == "high":
                st.warning("⚠️ **Require additional verification** (step-up MFA).")
            elif risk_level == "medium":
                st.info("ℹ️ **Monitor session** and apply rate limiting.")
            else:
                st.success("✅ **Allow login** - normal risk profile.")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")
            st.info("Make sure the API is running: `uvicorn api.main:app --reload`")

with tab2:
    st.header("Explain Risk Decision")
    st.markdown("Ask questions about risk decisions using AI (Phase 3 feature).")

    query = st.text_area(
        "Ask a question",
        value="Why was this login flagged as risky?",
        height=100,
        placeholder="e.g., Why was user X blocked? What factors contributed to the high score?"
    )
    login_id = st.text_input("Login ID (optional)", value="", placeholder="evt_000123")

    if st.button("🤖 Get Explanation", type="primary", use_container_width=True):
        payload = {
            "query": query,
            "login_id": login_id if login_id else None,
        }

        try:
            with st.spinner("Generating explanation..."):
                response = httpx.post(f"{api_url}/explain", json=payload, timeout=30.0)
                result = response.json()

            st.markdown("---")
            st.subheader("AI Explanation")

            st.info(result.get("explanation", "No explanation available"))

            confidence = result.get("confidence", 0)
            if confidence > 0:
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.0%}")

            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources")
                for source in sources:
                    st.markdown(f"- {source}")

        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.header("Dashboard")
    st.markdown("Real-time monitoring and analytics (Phase 4 feature).")

    # Placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logins", "10,000", "+342 today")
    col2.metric("High Risk", "127", "+8 today", delta_color="inverse")
    col3.metric("Avg Score", "0.12", "-0.02")
    col4.metric("Blocked", "23", "+3 today", delta_color="inverse")

    st.markdown("---")

    # Feature importance (from model training)
    st.subheader("Top Risk Indicators")
    feature_importance = {
        "New Device": 0.27,
        "IP Reputation": 0.26,
        "Device Age": 0.26,
        "Unusual Hour": 0.06,
        "Hour of Day": 0.06,
        "Location Changed": 0.05,
    }

    for feature, importance in feature_importance.items():
        col1, col2 = st.columns([3, 1])
        col1.write(feature)
        col2.progress(importance)

    st.markdown("---")
    st.info("📊 Full monitoring dashboard with Grafana integration coming in Phase 4.")

# Footer
st.markdown("---")
st.markdown(
    "**Identity Risk Agent Platform** | "
    "[GitHub](https://github.com/vijay-prabhu/identity-risk-agent) | "
    "Built with FastAPI + scikit-learn + Streamlit"
)
