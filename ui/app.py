"""
Identity Risk Agent - Streamlit Dashboard

Interactive UI for risk scoring and agent explanations.
"""

import streamlit as st
import httpx
import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Identity Risk Agent",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Identity Risk Agent")
st.markdown("ML-powered identity risk scoring with AI explanations")

# Sidebar
st.sidebar.header("Configuration")
tenant_id = st.sidebar.text_input("Tenant ID", value="default")
api_url = st.sidebar.text_input("API URL", value=API_URL)

# Main content
tab1, tab2, tab3 = st.tabs(["Score Login", "Explain Risk", "Dashboard"])

with tab1:
    st.header("Score Login Event")

    col1, col2 = st.columns(2)

    with col1:
        user_id = st.text_input("User ID", value="user123")
        ip_address = st.text_input("IP Address", value="192.168.1.1")
        device_id = st.text_input("Device ID", value="device_abc")

    with col2:
        location = st.text_input("Location", value="San Francisco, CA")
        mfa_used = st.checkbox("MFA Used")
        vpn_detected = st.checkbox("VPN Detected")

    if st.button("Score Login", type="primary"):
        payload = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "ip": ip_address,
            "device_id": device_id,
            "location": location,
            "mfa_used": mfa_used,
            "vpn_detected": vpn_detected,
        }

        try:
            response = httpx.post(f"{api_url}/score", json=payload)
            result = response.json()

            # Display risk score
            risk_score = result.get("risk_score", 0)
            risk_level = result.get("risk_level", "unknown")

            col1, col2, col3 = st.columns(3)
            col1.metric("Risk Score", f"{risk_score:.2f}")
            col2.metric("Risk Level", risk_level.upper())
            col3.metric("Factors", len(result.get("factors", [])))

            if result.get("factors"):
                st.subheader("Risk Factors")
                for factor in result["factors"]:
                    st.write(f"- {factor}")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")

with tab2:
    st.header("Explain Risk Decision")

    query = st.text_area(
        "Ask a question",
        value="Why was the login flagged as risky?",
        height=100,
    )
    login_id = st.text_input("Login ID (optional)", value="")

    if st.button("Get Explanation", type="primary"):
        payload = {
            "query": query,
            "login_id": login_id if login_id else None,
            "tenant_id": tenant_id,
        }

        try:
            response = httpx.post(f"{api_url}/explain", json=payload)
            result = response.json()

            st.subheader("Explanation")
            st.write(result.get("explanation", "No explanation available"))

            confidence = result.get("confidence", 0)
            st.progress(confidence, text=f"Confidence: {confidence:.0%}")

            if result.get("sources"):
                st.subheader("Sources")
                for source in result["sources"]:
                    st.write(f"- {source}")

        except Exception as e:
            st.error(f"Error connecting to API: {e}")

with tab3:
    st.header("Dashboard")
    st.info("Dashboard metrics will be implemented in Phase 4")

    # Placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logins", "0")
    col2.metric("High Risk", "0")
    col3.metric("Avg Score", "0.00")
    col4.metric("Agent Queries", "0")

# Footer
st.markdown("---")
st.markdown("Identity Risk Agent Platform | [GitHub](https://github.com/yourusername/identity-risk-agent)")
