import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
import joblib


def evaluate():
    # ── Load data & model ──────────────────────────────────
    df = pd.read_csv("data/engineered.csv")
    model = joblib.load("models/best_model.pkl")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Metrics ────────────────────────────────────────────
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # ── Risk Categorization ────────────────────────────────
    print("=" * 60)
    print("CUSTOMER RISK CATEGORIZATION")
    print("=" * 60)

    risk_df = X_test.copy()
    risk_df["churn_probability"] = y_prob
    risk_df["actual_churn"] = y_test.values

    def categorize_risk(prob):
        if prob >= 0.7:
            return "High Risk"
        elif prob >= 0.4:
            return "Medium Risk"
        else:
            return "Low Risk"

    risk_df["risk_category"] = risk_df["churn_probability"].apply(categorize_risk)
    print(risk_df["risk_category"].value_counts())
    print(
        f"\nHigh Risk Churn Rate:   {risk_df[risk_df['risk_category']=='High Risk']['actual_churn'].mean():.2%}"
    )
    print(
        f"Medium Risk Churn Rate: {risk_df[risk_df['risk_category']=='Medium Risk']['actual_churn'].mean():.2%}"
    )
    print(
        f"Low Risk Churn Rate:    {risk_df[risk_df['risk_category']=='Low Risk']['actual_churn'].mean():.2%}"
    )

    # ── Feature Importance ─────────────────────────────────
    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    print(importance_df.head(10))

    # ── Plots ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"AUC = {roc_auc_score(y_test, y_prob):.4f}",
    )
    axes[0].plot([0, 1], [0, 1], color="navy", linestyle="--")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=axes[1])
    axes[1].set_title("Confusion Matrix")

    # Feature Importance
    importance_df.head(10).plot(
        kind="barh",
        x="feature",
        y="importance",
        ax=axes[2],
        color="steelblue",
        legend=False,
    )
    axes[2].set_title("Top 10 Feature Importances")
    axes[2].invert_yaxis()

    plt.tight_layout()
    plt.savefig("data/evaluation_plots.png")
    print("\nPlots saved to data/evaluation_plots.png")
    plt.show()

    print("\n✅ Evaluation Complete!")
    return risk_df


if __name__ == "__main__":
    evaluate()
