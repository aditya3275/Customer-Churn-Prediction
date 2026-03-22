import streamlit as st
import requests
import os

st.set_page_config(page_title="Predict Churn", page_icon="🔍", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.title("🔍 Customer Churn Predictor")
st.markdown(
    "Enter customer details below to predict churn probability and risk category."
)
st.divider()

# ── Input Form ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    customer_id = st.text_input("Customer ID (leave blank to auto-generate)", value="")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📡 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

with col3:
    st.subheader("💳 Account")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.divider()

# ── Predict ────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):
    payload = {
        "customer_id":    customer_id.strip() if customer_id.strip() else None,
        "gender":          gender,
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "MultipleLines":   multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity":  online_security,
        "OnlineBackup":    online_backup,
        "DeviceProtection": device_protection,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract":        contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":   payment,
        "MonthlyCharges":  monthly_charges,
        "TotalCharges":    total_charges,
    }

    try:
        response = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("🎯 Prediction Results")

        # ── DB confirmation badge ──────────────────────────────
        st.success(
            f"✅ **Prediction saved to database** — "
            f"ID `{result['prediction_id']}` · "
            f"Customer `{result['customer_id']}`"
        )

        col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)

        with col_a:
            churn = result["churn_prediction"]
            color = "🔴" if churn == "Yes" else "🟢"
            st.metric("Churn Prediction", f"{color} {churn}")

        with col_b:
            st.metric("Churn Probability", result["churn_probability"])

        with col_c:
            confidence_pct = f"{round(result['confidence_score'] * 100, 1)}%"
            st.metric("Confidence", confidence_pct)

        with col_d:
            risk = result["risk_category"]
            risk_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
            st.metric("Risk Category", f"{risk_color[risk]} {risk}")

        with col_e:
            st.metric("Expected CLV", f"${result['lifetime_value']:,.2f}")

        with col_f:
            st.metric("Prediction ID", f"#{result['prediction_id']}")

        # ── Probability Meter ──────────────────────────────────
        st.divider()
        st.subheader("📈 Churn Probability Meter")
        prob = result["churn_probability"]
        st.progress(prob)

        # ── Alert ──────────────────────────────────────────────
        if risk == "High Risk":
            st.error("⚠️ HIGH RISK — Immediate retention action recommended!")
        elif risk == "Medium Risk":
            st.warning("⚡ MEDIUM RISK — Consider a retention offer soon.")
        else:
            st.success("✅ LOW RISK — Customer is likely to stay.")

        # ── Retention Tips ─────────────────────────────────────
        st.divider()
        st.subheader("💡 Recommended Retention Actions")

        if contract == "Month-to-month":
            st.warning("📋 Offer a discounted One Year or Two Year contract upgrade")
        if internet_service == "Fiber optic" and online_security == "No":
            st.warning(
                "🔒 Bundle Online Security — fiber customers churn when they feel unsafe"
            )
        if payment == "Electronic check":
            st.warning("💳 Encourage switch to Auto-pay — reduces payment friction")
        if tenure < 6:
            st.warning(
                "🎁 New customer detected — offer a loyalty bonus or free service upgrade"
            )
        if prob < 0.4:
            st.success("🌟 Customer is satisfied — consider upselling premium services")

        # ── History shortcut ───────────────────────────────────
        st.divider()
        cid = result["customer_id"]
        if st.button(f"📂 View full history for customer {cid}"):
            with st.spinner("Fetching history…"):
                hist = requests.get(
                    f"{API_BASE}/customer-history/{cid}", timeout=10
                ).json()
            st.json(hist)

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot reach the API. Make sure FastAPI is running:\n```\nuvicorn main:app --reload\n```")
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.divider()
st.caption("ChurnGuard AI — Powered by Gradient Boosting | AUC: 0.8336 | DB-backed v2.0")
