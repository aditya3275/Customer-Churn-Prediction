import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ──────────────────────────────────────────────
df = pd.read_csv("data/telco_churn.csv")

# ── Basic Info ─────────────────────────────────────────────
print("=" * 50)
print("SHAPE:", df.shape)
print("=" * 50)
print("\nCOLUMNS:\n", df.columns.tolist())
print("\nDTYPES:\n", df.dtypes)
print("\nFIRST 5 ROWS:\n", df.head())

# ── Missing Values ─────────────────────────────────────────
print("\n" + "=" * 50)
print("MISSING VALUES:")
print(df.isnull().sum())
print("\nMISSING %:")
print((df.isnull().sum() / len(df)) * 100)

# ── Target Distribution ────────────────────────────────────
print("\n" + "=" * 50)
print("CHURN DISTRIBUTION:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# ── Numerical Summary ──────────────────────────────────────
print("\n" + "=" * 50)
print("NUMERICAL SUMMARY:")
print(df.describe())

# ── Plots ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Churn count
df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["steelblue", "tomato"])
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")

# Tenure distribution
df["tenure"].hist(bins=30, ax=axes[1], color="steelblue", edgecolor="black")
axes[1].set_title("Tenure Distribution")
axes[1].set_xlabel("Tenure (months)")

plt.tight_layout()
plt.savefig("data/eda_plots.png")
print("\nPlot saved to data/eda_plots.png")
plt.show()

print("\n✅ EDA Complete!")
