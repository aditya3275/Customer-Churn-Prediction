import streamlit as st

st.set_page_config(page_title="ChurnGuard AI", page_icon="🛡️", layout="wide")

st.title("🛡️ ChurnGuard AI")
st.subheader("Intelligent Customer Churn Prevention System")
st.divider()

st.markdown(
    """
Welcome to **ChurnGuard AI** — a complete machine learning system to detect,
analyze, and prevent customer churn.

### 📌 Navigate using the sidebar:

| Page | Description |
|---|---|
| 🔍 Predict | Predict churn for a single customer |
| 📊 Dashboard | Full churn risk overview across all customers |
| ⚠️ Early Warning | Customers showing early churn signals |
| 👥 Segmentation | High risk customer groups and reasons |
| 💰 CLV | Customer Lifetime Value prioritization + retention strategies |

---
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.info("🎯 Model: Gradient Boosting")
with col2:
    st.info("📈 AUC-ROC: 0.8336")
with col3:
    st.info("👥 Customers Analyzed: 7,043")
