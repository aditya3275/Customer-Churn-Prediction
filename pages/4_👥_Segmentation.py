import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")

st.title("👥 High Risk Customer Segmentation")
st.markdown(
    "Grouping high risk customers by churn reason to enable targeted retention."
)
st.divider()


# ── Load Data ──────────────────────────────────────────────
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
high_risk_df = df[df["risk_category"] == "High Risk"].copy()

# ── KPI ────────────────────────────────────────────────────
st.subheader("📌 High Risk Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🔴 Total High Risk", f"{len(high_risk_df):,}")
with col2:
    st.metric("💰 Avg Monthly Charges", f"${high_risk_df['MonthlyCharges'].mean():.2f}")
with col3:
    st.metric("📅 Avg Tenure", f"{high_risk_df['tenure'].mean():.1f} months")
with col4:
    st.metric(
        "☠️ Avg Churn Probability", f"{high_risk_df['churn_probability'].mean():.2%}"
    )

st.divider()


# ── Segment Definition ─────────────────────────────────────
def assign_segment(row):
    if row["Contract"] == "Month-to-month" and row["tenure"] <= 12:
        return "🆕 New & Uncommitted"
    elif row["InternetService"] == "Fiber optic" and row["MonthlyCharges"] > 80:
        return "💸 High Paying Fiber Users"
    elif row["OnlineSecurity"] == "No" and row["TechSupport"] == "No":
        return "🔓 Unprotected Users"
    elif (
        row["PaymentMethod"] == "Electronic check"
        and row["Contract"] == "Month-to-month"
    ):
        return "💳 Payment Risk Group"
    elif row["SeniorCitizen"] == 1:
        return "👴 Senior Citizens"
    else:
        return "📦 General High Risk"


high_risk_df["segment"] = high_risk_df.apply(assign_segment, axis=1)

# ── Segment Distribution ───────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🧩 Segment Distribution")
    fig, ax = plt.subplots(figsize=(6, 5))
    seg_counts = high_risk_df["segment"].value_counts()
    colors = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c", "#f39c12"]
    ax.pie(
        seg_counts,
        labels=seg_counts.index,
        colors=colors[: len(seg_counts)],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("High Risk Customer Segments")
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("📊 Avg Churn Probability by Segment")
    fig, ax = plt.subplots(figsize=(6, 5))
    seg_prob = high_risk_df.groupby("segment")["churn_probability"].mean().sort_values()
    seg_prob.plot(kind="barh", ax=ax, color="tomato", edgecolor="black")
    ax.set_title("Avg Churn Probability per Segment")
    ax.set_xlabel("Avg Churn Probability")
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Segment Deep Dive ──────────────────────────────────────
st.subheader("🔎 Segment Deep Dive")
selected_segment = st.selectbox(
    "Select a Segment to Explore", high_risk_df["segment"].unique()
)

seg_df = high_risk_df[high_risk_df["segment"] == selected_segment]

col_c, col_d, col_e = st.columns(3)
with col_c:
    st.metric("👥 Customers", f"{len(seg_df):,}")
with col_d:
    st.metric("💰 Avg Monthly Charges", f"${seg_df['MonthlyCharges'].mean():.2f}")
with col_e:
    st.metric("☠️ Avg Churn Probability", f"{seg_df['churn_probability'].mean():.2%}")

# ── Segment Charts ─────────────────────────────────────────
col_f, col_g = st.columns(2)

with col_f:
    st.subheader("📋 Contract Mix")
    fig, ax = plt.subplots(figsize=(5, 3))
    seg_df["Contract"].value_counts().plot(
        kind="bar", ax=ax, color="steelblue", edgecolor="black"
    )
    ax.set_title("Contract Types")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

with col_g:
    st.subheader("🌐 Internet Service Mix")
    fig, ax = plt.subplots(figsize=(5, 3))
    seg_df["InternetService"].value_counts().plot(
        kind="bar", ax=ax, color="mediumpurple", edgecolor="black"
    )
    ax.set_title("Internet Service Types")
    plt.xticks(rotation=30)
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Segment Retention Strategy ─────────────────────────────
st.subheader("💡 Recommended Retention Strategy for This Segment")

strategies = {
    "🆕 New & Uncommitted": [
        "🎁 Offer a welcome loyalty bonus after 3 months",
        "📋 Provide discounted upgrade to One Year contract",
        "📞 Schedule a proactive customer success call in month 2",
        "🎯 Send personalized onboarding tips to increase engagement",
    ],
    "💸 High Paying Fiber Users": [
        "🔒 Bundle Online Security and Tech Support for free for 3 months",
        "💰 Offer a loyalty discount on monthly charges",
        "⚡ Highlight service reliability and speed advantages",
        "🎁 Reward long-term commitment with premium perks",
    ],
    "🔓 Unprotected Users": [
        "🔒 Offer Online Security at 50% off for first 6 months",
        "🔧 Bundle Tech Support with current plan",
        "📧 Send educational content about online safety risks",
        "🛡️ Promote device protection add-on",
    ],
    "💳 Payment Risk Group": [
        "💳 Incentivize switch to automatic payment with a discount",
        "📋 Offer contract upgrade to reduce monthly commitment anxiety",
        "🔔 Set up payment reminders to reduce friction",
        "💰 Offer a small bill credit for switching to auto-pay",
    ],
    "👴 Senior Citizens": [
        "📞 Assign a dedicated customer support representative",
        "🎁 Offer senior loyalty discount program",
        "🔧 Provide free tech support sessions",
        "📱 Simplify billing and payment process",
    ],
    "📦 General High Risk": [
        "📞 Proactive outreach call from retention team",
        "💰 Offer personalized discount based on usage",
        "📋 Suggest contract upgrade with incentive",
        "🎯 Send targeted satisfaction survey and act on feedback",
    ],
}

for strategy in strategies.get(selected_segment, []):
    st.info(strategy)

st.divider()

# ── Full Segment Table ─────────────────────────────────────
st.subheader("🗃️ Full Segment Customer Table")
display_df = seg_df[
    [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "InternetService",
        "PaymentMethod",
        "churn_probability",
        "segment",
    ]
].copy()
display_df["churn_probability"] = display_df["churn_probability"].round(4)
display_df = display_df.sort_values("churn_probability", ascending=False)
st.dataframe(display_df, use_container_width=True)
st.caption(f"Showing {len(display_df)} customers in {selected_segment} segment")

st.divider()
st.caption("ChurnGuard AI — Powered by Gradient Boosting | AUC: 0.8336")
