import streamlit as st
import requests
import json

# ── Page Config ────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

# ── Header ─────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn and their risk category.")
st.divider()

# ── Input Form ─────────────────────────────────────────────
st.subheader("🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
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

# ── Predict Button ─────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        result = response.json()

        st.divider()
        st.subheader("🎯 Prediction Results")

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            churn = result["churn_prediction"]
            color = "🔴" if churn == "Yes" else "🟢"
            st.metric("Churn Prediction", f"{color} {churn}")

        with col_b:
            st.metric("Churn Probability", result["churn_probability"])

        with col_c:
            st.metric("Confidence", result["confidence"])

        with col_d:
            risk = result["risk_category"]
            risk_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
            st.metric("Risk Category", f"{risk_color[risk]} {risk}")

        # ── Risk bar ───────────────────────────────────────
        st.divider()
        prob = result["churn_probability"]
        st.subheader("📈 Churn Probability Meter")
        st.progress(prob)

        if risk == "High Risk":
            st.error(
                "⚠️ This customer is at HIGH RISK of churning. Immediate retention action recommended!"
            )
        elif risk == "Medium Risk":
            st.warning(
                "⚡ This customer is at MEDIUM RISK. Consider a retention offer."
            )
        else:
            st.success("✅ This customer is at LOW RISK. Likely to stay!")

    except Exception as e:
        st.error(
            f"❌ Could not connect to API. Make sure FastAPI is running!\nError: {e}"
        )

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.markdown("Built with ❤️ using Streamlit + FastAPI + Gradient Boosting")
