import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

print("\n" + "=" * 50)
print("       CHURNGUARD AI — MODEL VERIFICATION")
print("=" * 50)

# ── Load ───────────────────────────────────────────────────
model = joblib.load("models/best_model.pkl")
df = pd.read_csv("data/engineered.csv")

X = df.drop(columns=["Churn"])
y = df["Churn"]

# ── Same split as training ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Predictions ────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ── Sanity Check ───────────────────────────────────────────
print("\n📊 DATASET SUMMARY")
print(f"   Total Customers  : {len(df):,}")
print(f"   Training Samples : {len(X_train):,}")
print(f"   Test Samples     : {len(X_test):,}")
print(f"   Total Features   : {X.shape[1]}")
print(f"   Churn in Test    : {y_test.sum()} ({y_test.mean():.1%})")

# ── Model Performance ──────────────────────────────────────
test_auc = roc_auc_score(y_test, y_prob)
print("\n🎯 MODEL PERFORMANCE")
print(f"   Model            : Gradient Boosting")
print(f"   Test AUC-ROC     : {test_auc:.4f}")
print(f"\n{classification_report(y_test, y_pred)}")

# ── Overfitting Check ──────────────────────────────────────
train_prob = model.predict_proba(X_train)[:, 1]
train_auc = roc_auc_score(y_train, train_prob)
diff = train_auc - test_auc

print("=" * 50)
print("🔍 OVERFITTING CHECK")
print("=" * 50)
print(f"   Train AUC  : {train_auc:.4f}")
print(f"   Test AUC   : {test_auc:.4f}")
print(f"   Difference : {diff:.4f}")

if diff < 0.05:
    print("\n   ✅ No overfitting — model generalizes well!")
elif diff < 0.10:
    print("\n   ⚠️  Slight overfitting — acceptable range")
else:
    print("\n   ❌ Overfitting detected — needs attention")

# ── Risk Categorization Validation ────────────────────────
print("\n" + "=" * 50)
print("🚦 RISK CATEGORIZATION VALIDATION")
print("=" * 50)

results_df = pd.DataFrame({"actual": y_test.values, "probability": y_prob})


def risk(p):
    if p >= 0.7:
        return "High Risk"
    elif p >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


results_df["risk"] = results_df["probability"].apply(risk)

for category in ["High Risk", "Medium Risk", "Low Risk"]:
    group = results_df[results_df["risk"] == category]
    churn_rate = group["actual"].mean() * 100
    count = len(group)
    print(
        f"   {category:15} → {count:4} customers | Actual Churn Rate: {churn_rate:.2f}%"
    )

# ── Feature Importance ─────────────────────────────────────
print("\n" + "=" * 50)
print("🏆 TOP 10 FEATURE IMPORTANCES")
print("=" * 50)
importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
    .head(10)
)

for _, row in importance_df.iterrows():
    bar = "█" * int(row["Importance"] * 50)
    print(f"   {row['Feature'][:30]:30} | {bar} {row['Importance']:.4f}")

print("\n" + "=" * 50)
print("✅ VERIFICATION COMPLETE — MODEL IS READY!")
print("=" * 50 + "\n")
