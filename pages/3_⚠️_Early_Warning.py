import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Early Warning System", page_icon="⚠️", layout="wide")

st.title("⚠️ Early Warning System")
st.markdown("Customers showing early churn signals before it's too late.")
st.divider()


# ── Load Data & Model ──────────────────────────────────────
@st.cache_data
def load_data():
    df_raw = pd.read_csv("data/telco_churn.csv")
    df_eng = pd.read_csv("data/engineered.csv")
    model = joblib.load("models/best_model.pkl")

    X = df_eng.drop(columns=["Churn"])
    y = df_eng["Churn"]

    probs = model.predict_proba(X)[:, 1]
    df_raw["churn_probability"] = probs
    df_raw["actual_churn"] = y.values

    def risk(p):
        if p >= 0.7:
            return "High Risk"
        elif p >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    df_raw["risk_category"] = df_raw["churn_probability"].apply(risk)
    return df_raw


df = load_data()

# ── Early Warning Signals ──────────────────────────────────
st.subheader("🚨 Early Warning Signal Detection")
st.markdown("Customers are flagged based on multiple early churn signals:")


# Define warning signals
def get_warnings(row):
    warnings = []
    if row["tenure"] <= 6:
        warnings.append("🆕 New Customer (≤6 months)")
    if row["Contract"] == "Month-to-month":
        warnings.append("📋 Month-to-month Contract")
    if row["PaymentMethod"] == "Electronic check":
        warnings.append("💳 Electronic Check Payment")
    if row["InternetService"] == "Fiber optic":
        warnings.append("🌐 Fiber Optic (High Cost Service)")
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        warnings.append("🔓 No Online Security")
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        warnings.append("🔧 No Tech Support")
    if row["MonthlyCharges"] > 80:
        warnings.append("💰 High Monthly Charges (>$80)")
    if row["SeniorCitizen"] == 1:
        warnings.append("👴 Senior Citizen")
    return warnings


df["warnings"] = df.apply(get_warnings, axis=1)
df["warning_count"] = df["warnings"].apply(len)

# ── KPI Metrics ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

critical = len(df[df["warning_count"] >= 4])
high_warn = len(df[df["warning_count"] == 3])
medium_warn = len(df[df["warning_count"] == 2])
low_warn = len(df[df["warning_count"] <= 1])

with col1:
    st.metric("🚨 Critical (4+ signals)", f"{critical:,}")
with col2:
    st.metric("🔴 High (3 signals)", f"{high_warn:,}")
with col3:
    st.metric("🟡 Medium (2 signals)", f"{medium_warn:,}")
with col4:
    st.metric("🟢 Low (0-1 signals)", f"{low_warn:,}")

st.divider()

# ── Warning Count Distribution ─────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Warning Signal Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    warn_counts = df["warning_count"].value_counts().sort_index()
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#c0392b"]
    warn_counts.plot(
        kind="bar", ax=ax, color=colors[: len(warn_counts)], edgecolor="black"
    )
    ax.set_title("Customers by Number of Warning Signals")
    ax.set_xlabel("Number of Warning Signals")
    ax.set_ylabel("Number of Customers")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("📈 Warning Count vs Churn Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    warn_prob = df.groupby("warning_count")["churn_probability"].mean()
    warn_prob.plot(
        kind="line", ax=ax, marker="o", color="tomato", linewidth=2, markersize=8
    )
    ax.set_title("Avg Churn Probability by Warning Count")
    ax.set_xlabel("Number of Warning Signals")
    ax.set_ylabel("Avg Churn Probability")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Critical Customers Table ───────────────────────────────
st.subheader("🚨 Customers Needing Immediate Attention")

min_warnings = st.slider("Show customers with at least N warning signals", 1, 8, 4)

critical_df = df[df["warning_count"] >= min_warnings].copy()
critical_df = critical_df.sort_values("churn_probability", ascending=False)

display_df = critical_df[
    [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "InternetService",
        "churn_probability",
        "risk_category",
        "warning_count",
    ]
].copy()
display_df["churn_probability"] = display_df["churn_probability"].round(4)
display_df["warnings"] = critical_df["warnings"].apply(lambda x: " | ".join(x))

st.dataframe(display_df, use_container_width=True)
st.caption(
    f"⚠️ {len(display_df)} customers flagged with {min_warnings}+ warning signals"
)

st.divider()

# ── Signal Frequency ───────────────────────────────────────
st.subheader("📌 Most Common Warning Signals")

all_warnings = []
for w_list in df["warnings"]:
    all_warnings.extend(w_list)

warn_series = pd.Series(all_warnings).value_counts()

fig, ax = plt.subplots(figsize=(10, 4))
warn_series.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
ax.set_title("Frequency of Each Warning Signal Across All Customers")
ax.set_xlabel("Number of Customers")
ax.invert_yaxis()
st.pyplot(fig)
plt.close()

st.divider()
st.caption("ChurnGuard AI — Powered by Gradient Boosting | AUC: 0.8336")
