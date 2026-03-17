import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="CLV & Retention", page_icon="💰", layout="wide")

st.title("💰 Customer Lifetime Value & Retention Strategy")
st.markdown(
    "Prioritize retention efforts based on Customer Lifetime Value and churn risk."
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

    # ── CLV Formula ────────────────────────────────────────
    # CLV = MonthlyCharges × Tenure × (1 - ChurnProbability)
    df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
    df_raw["TotalCharges"] = df_raw["TotalCharges"].fillna(
        df_raw["TotalCharges"].median()
    )
    df_raw["CLV"] = (
        df_raw["MonthlyCharges"]
        * (df_raw["tenure"] + 1)
        * (1 - df_raw["churn_probability"])
    )
    df_raw["CLV"] = df_raw["CLV"].round(2)

    # ── CLV Tier ───────────────────────────────────────────
    clv_75 = df_raw["CLV"].quantile(0.75)
    clv_40 = df_raw["CLV"].quantile(0.40)

    def clv_tier(clv):
        if clv >= clv_75:
            return "💎 Platinum"
        elif clv >= clv_40:
            return "🥇 Gold"
        else:
            return "🥈 Silver"

    df_raw["CLV_tier"] = df_raw["CLV"].apply(clv_tier)

    return df_raw


df = load_data()

# ── KPI Metrics ────────────────────────────────────────────
st.subheader("📌 CLV Overview")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("💎 Total CLV", f"${df['CLV'].sum():,.0f}")
with col2:
    st.metric("📊 Avg CLV", f"${df['CLV'].mean():,.2f}")
with col3:
    st.metric("💎 Platinum Customers", f"{len(df[df['CLV_tier'] == '💎 Platinum']):,}")
with col4:
    st.metric("🥇 Gold Customers", f"{len(df[df['CLV_tier'] == '🥇 Gold']):,}")
with col5:
    at_risk_clv = df[df["risk_category"] == "High Risk"]["CLV"].sum()
    st.metric("🔴 CLV at Risk", f"${at_risk_clv:,.0f}")

st.divider()

# ── Charts Row 1 ───────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("💎 CLV Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["CLV"], bins=40, color="gold", edgecolor="black", alpha=0.8)
    ax.set_title("Customer Lifetime Value Distribution")
    ax.set_xlabel("CLV ($)")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)
    plt.close()

with col_b:
    st.subheader("🎯 CLV by Risk Category")
    fig, ax = plt.subplots(figsize=(6, 4))
    clv_risk = df.groupby("risk_category")["CLV"].mean().sort_values()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    clv_risk.plot(kind="barh", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Avg CLV by Risk Category")
    ax.set_xlabel("Average CLV ($)")
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Charts Row 2 ───────────────────────────────────────────
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("📊 CLV Tier Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    tier_counts = df["CLV_tier"].value_counts()
    colors = ["#f1c40f", "#95a5a6", "#f39c12"]
    ax.pie(
        tier_counts,
        labels=tier_counts.index,
        colors=colors[: len(tier_counts)],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Customer CLV Tier Distribution")
    st.pyplot(fig)
    plt.close()

with col_d:
    st.subheader("📈 CLV vs Churn Probability")
    fig, ax = plt.subplots(figsize=(6, 4))
    colors_map = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}
    for risk_cat, group in df.groupby("risk_category"):
        ax.scatter(
            group["churn_probability"],
            group["CLV"],
            alpha=0.3,
            label=risk_cat,
            color=colors_map[risk_cat],
            s=10,
        )
    ax.set_xlabel("Churn Probability")
    ax.set_ylabel("CLV ($)")
    ax.set_title("CLV vs Churn Probability")
    ax.legend()
    st.pyplot(fig)
    plt.close()

st.divider()

# ── Priority Retention Table ───────────────────────────────
st.subheader("🎯 Priority Retention List")
st.markdown(
    "High risk customers sorted by CLV — these are the most valuable customers we cannot afford to lose!"
)

priority_df = df[df["risk_category"] == "High Risk"].copy()
priority_df = priority_df.sort_values("CLV", ascending=False)

display_df = priority_df[
    [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "churn_probability",
        "CLV",
        "CLV_tier",
        "risk_category",
    ]
].copy()
display_df["churn_probability"] = display_df["churn_probability"].round(4)

st.dataframe(display_df.head(50), use_container_width=True)
st.caption(f"Top 50 of {len(priority_df)} high risk customers by CLV")

st.divider()

# ── Retention Strategy Builder ─────────────────────────────
st.subheader("🛡️ Retention Strategy Builder")
st.markdown("Select a CLV tier and risk level to get a targeted retention strategy:")

col_e, col_f = st.columns(2)
with col_e:
    selected_tier = st.selectbox(
        "Select CLV Tier", ["💎 Platinum", "🥇 Gold", "🥈 Silver"]
    )
with col_f:
    selected_risk = st.selectbox(
        "Select Risk Category", ["High Risk", "Medium Risk", "Low Risk"]
    )

st.divider()

# ── Strategy Matrix ────────────────────────────────────────
strategies = {
    ("💎 Platinum", "High Risk"): {
        "priority": "🚨 CRITICAL — Act within 24 hours",
        "actions": [
            "📞 Personal call from senior retention manager",
            "💰 Offer 30% discount on monthly charges for 6 months",
            "🎁 Free upgrade to premium plan with all services included",
            "📋 Propose Two Year contract with significant price lock",
            "🌟 Assign dedicated VIP account manager",
        ],
    },
    ("💎 Platinum", "Medium Risk"): {
        "priority": "⚠️ HIGH PRIORITY — Act within 48 hours",
        "actions": [
            "📞 Proactive outreach call from retention team",
            "💰 Offer 15% loyalty discount on next 3 months",
            "🎁 Free add-on service for 3 months",
            "📋 Suggest One Year contract upgrade with price guarantee",
        ],
    },
    ("💎 Platinum", "Low Risk"): {
        "priority": "✅ MAINTAIN — Nurture relationship",
        "actions": [
            "🌟 Enroll in VIP loyalty rewards program",
            "🎁 Surprise and delight — send appreciation gift",
            "📈 Upsell premium add-on services",
            "📞 Annual satisfaction check-in call",
        ],
    },
    ("🥇 Gold", "High Risk"): {
        "priority": "🔴 URGENT — Act within 48 hours",
        "actions": [
            "📞 Retention call with personalized offer",
            "💰 Offer 20% discount for committing to One Year contract",
            "🎁 Free Tech Support or Online Security for 3 months",
            "📋 Highlight value of current plan vs competitors",
        ],
    },
    ("🥇 Gold", "Medium Risk"): {
        "priority": "🟡 MONITOR — Act within 1 week",
        "actions": [
            "📧 Send personalized email with loyalty offer",
            "💰 Offer a small bill credit as appreciation",
            "📋 Suggest contract upgrade with minor incentive",
            "🔔 Enroll in proactive service update notifications",
        ],
    },
    ("🥇 Gold", "Low Risk"): {
        "priority": "✅ MAINTAIN — Focus on upselling",
        "actions": [
            "📈 Offer premium service bundle upgrade",
            "🎁 Referral bonus program enrollment",
            "📞 Bi-annual satisfaction survey",
            "🌟 Loyalty points or rewards program",
        ],
    },
    ("🥈 Silver", "High Risk"): {
        "priority": "🟠 IMPORTANT — Act within 1 week",
        "actions": [
            "📧 Automated personalized retention email",
            "💰 Offer modest discount for contract upgrade",
            "📋 Highlight plan benefits and value",
            "🎁 Free trial of one premium add-on",
        ],
    },
    ("🥈 Silver", "Medium Risk"): {
        "priority": "🟡 LOW PRIORITY — Automated outreach",
        "actions": [
            "📧 Automated email with satisfaction survey",
            "🔔 Enroll in service improvement notifications",
            "📋 Share tips to get more value from current plan",
        ],
    },
    ("🥈 Silver", "Low Risk"): {
        "priority": "✅ STABLE — Minimal intervention needed",
        "actions": [
            "📧 Monthly newsletter with product updates",
            "📈 Occasional upsell opportunity emails",
            "🌟 Invite to refer friends for rewards",
        ],
    },
}

strategy = strategies.get((selected_tier, selected_risk), {})

if strategy:
    if "CRITICAL" in strategy["priority"] or "URGENT" in strategy["priority"]:
        st.error(f"**Priority Level:** {strategy['priority']}")
    elif "HIGH" in strategy["priority"] or "IMPORTANT" in strategy["priority"]:
        st.warning(f"**Priority Level:** {strategy['priority']}")
    else:
        st.success(f"**Priority Level:** {strategy['priority']}")

    st.subheader("📋 Recommended Actions:")
    for action in strategy["actions"]:
        st.info(action)

st.divider()

# ── Revenue Impact ─────────────────────────────────────────
st.subheader("📉 Revenue Impact Analysis")

col_g, col_h, col_i = st.columns(3)

total_clv_at_risk = df[df["risk_category"] == "High Risk"]["CLV"].sum()
avg_retention_cost = 50
customers_to_save = len(df[df["risk_category"] == "High Risk"])
potential_saving = total_clv_at_risk * 0.6

with col_g:
    st.metric("💸 Total CLV at Risk", f"${total_clv_at_risk:,.0f}")
with col_h:
    st.metric("🎯 Potential Revenue Saved (60%)", f"${potential_saving:,.0f}")
with col_i:
    st.metric(
        "💰 Estimated Retention Budget",
        f"${customers_to_save * avg_retention_cost:,.0f}",
    )

st.divider()
st.caption("ChurnGuard AI — Powered by Gradient Boosting | AUC: 0.8336")
