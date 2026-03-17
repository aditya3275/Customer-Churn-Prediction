import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

st.title("📊 Churn Risk Dashboard")
st.markdown("Complete overview of customer churn risk across the entire dataset.")
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

# ── KPI Metrics ────────────────────────────────────────────
st.subheader("📌 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

total = len(df)
high_risk = len(df[df["risk_category"] == "High Risk"])
medium_risk = len(df[df["risk_category"] == "Medium Risk"])
low_risk = len(df[df["risk_category"] == "Low Risk"])
avg_prob = df["churn_probability"].mean()

with col1:
    st.metric("Total Customers", f"{total:,}")
with col2:
    st.metric("🔴 High Risk", f"{high_risk:,}", f"{high_risk/total*100:.1f}%")
with col3:
    st.metric("🟡 Medium Risk", f"{medium_risk:,}", f"{medium_risk/total*100:.1f}%")
with col4:
    st.metric("🟢 Low Risk", f"{low_risk:,}", f"{low_risk/total*100:.1f}%")
with col5:
    st.metric("Avg Churn Probability", f"{avg_prob:.2%}")

st.divider()

# ── Charts Row 1 ───────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🎯 Risk Category Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    risk_counts = df["risk_category"].value_counts()
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax.pie(
        risk_counts,
        labels=risk_counts.index,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Customer Risk Distribution")
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("📊 Churn Probability Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        df["churn_probability"],
        bins=40,
        color="steelblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax.axvline(0.4, color="orange", linestyle="--", label="Medium Risk threshold")
    ax.axvline(0.7, color="red", linestyle="--", label="High Risk threshold")
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("Number of Customers")
    ax.set_title("Churn Probability Distribution")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Charts Row 2 ───────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("📋 Churn Rate by Contract Type")
    fig, ax = plt.subplots(figsize=(6, 4))
    contract_churn = df.groupby("Contract")["actual_churn"].mean() * 100
    contract_churn.plot(
        kind="bar", ax=ax, color=["#e74c3c", "#f39c12", "#2ecc71"], edgecolor="black"
    )
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

with col_d:
    st.subheader("💳 Churn Rate by Payment Method")
    fig, ax = plt.subplots(figsize=(6, 4))
    payment_churn = df.groupby("PaymentMethod")["actual_churn"].mean() * 100
    payment_churn.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title("Churn Rate by Payment Method")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Charts Row 3 ───────────────────────────────────────────
col_e, col_f = st.columns(2)

with col_e:
    st.subheader("📅 Churn Rate by Tenure Group")
    fig, ax = plt.subplots(figsize=(6, 4))
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "12-24m", "24-48m", "48-72m"],
    )
    tenure_churn = (
        df.groupby("tenure_group", observed=True)["actual_churn"].mean() * 100
    )
    tenure_churn.plot(kind="bar", ax=ax, color="coral", edgecolor="black")
    ax.set_title("Churn Rate by Tenure Group")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Group")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

with col_f:
    st.subheader("🌐 Churn Rate by Internet Service")
    fig, ax = plt.subplots(figsize=(6, 4))
    internet_churn = df.groupby("InternetService")["actual_churn"].mean() * 100
    internet_churn.plot(kind="bar", ax=ax, color="mediumpurple", edgecolor="black")
    ax.set_title("Churn Rate by Internet Service")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Internet Service")
    plt.xticks(rotation=0)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Raw Data Table ─────────────────────────────────────────
st.subheader("🗃️ Customer Risk Table")
filter_risk = st.selectbox(
    "Filter by Risk Category", ["All", "High Risk", "Medium Risk", "Low Risk"]
)

display_df = df[
    [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "InternetService",
        "PaymentMethod",
        "churn_probability",
        "risk_category",
    ]
].copy()
display_df["churn_probability"] = display_df["churn_probability"].round(4)

if filter_risk != "All":
    display_df = display_df[display_df["risk_category"] == filter_risk]

st.dataframe(display_df, use_container_width=True)
st.caption(f"Showing {len(display_df)} customers")

st.divider()
st.caption("ChurnGuard AI — Powered by Gradient Boosting | AUC: 0.8336")
