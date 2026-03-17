import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def engineer_features(filepath="data/processed.csv"):

    df = pd.read_csv(filepath)

    # ── Feature 1: Charge per Tenure Month ────────────────
    # How much is the customer paying per month relative to how long they've stayed
    df["charge_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # ── Feature 2: Is New Customer ────────────────────────
    # Customers with tenure < 6 months are high risk
    df["is_new_customer"] = (df["tenure"] < -0.5).astype(int)

    # ── Feature 3: Has Premium Services ───────────────────
    # Customers with more services are less likely to churn
    service_cols = [
        "OnlineSecurity_Yes",
        "OnlineBackup_Yes",
        "DeviceProtection_Yes",
        "TechSupport_Yes",
        "StreamingTV_Yes",
        "StreamingMovies_Yes",
    ]
    df["total_services"] = df[service_cols].sum(axis=1)

    # ── Feature 4: Is High Value Customer ─────────────────
    df["is_high_value"] = (df["MonthlyCharges"] > 0.5).astype(int)

    # ── Feature 5: Contract Risk Score ────────────────────
    # Month-to-month contracts are highest churn risk
    df["contract_risk"] = 1.0  # default month-to-month = high risk
    df.loc[df["Contract_One year"] == 1, "contract_risk"] = 0.5
    df.loc[df["Contract_Two year"] == 1, "contract_risk"] = 0

    # ── Save engineered data ───────────────────────────────
    df.to_csv("data/engineered.csv", index=False)
    print(f"✅ Feature Engineering Complete! Shape: {df.shape}")
    print(
        f"New features added: charge_per_tenure, is_new_customer, total_services, is_high_value, contract_risk"
    )

    # ── Correlation with Churn ─────────────────────────────
    print("\nTop correlations with Churn:")
    corr = df.corr()["Churn"].sort_values(ascending=False)
    print(corr.head(10))
    print("\nBottom correlations with Churn:")
    print(corr.tail(5))

    # ── Plot correlation heatmap ───────────────────────────
    plt.figure(figsize=(10, 8))
    top_features = corr.abs().sort_values(ascending=False).head(15).index
    sns.heatmap(df[top_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Top Feature Correlations")
    plt.tight_layout()
    plt.savefig("data/correlation_heatmap.png")
    print("\nHeatmap saved to data/correlation_heatmap.png")

    return df


if __name__ == "__main__":
    df = engineer_features()
