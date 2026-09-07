"""
Customer Churn Analysis — IBM Telco Customer Churn Dataset (real, via Kaggle)
------------------------------------------------------------------
Goes from raw data -> cleaning -> EDA -> a predictive model -> a ranked
at-risk customer list.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, RocCurveDisplay

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# ---------------------------------------------------------------
# Data cleaning: TotalCharges is stored as text, with 11 blank
# values for customers with 0 months tenure (brand new signups).
# These aren't errors, they're customers who haven't been billed yet.
# ---------------------------------------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_missing = df["TotalCharges"].isna().sum()
print(f"Found {n_missing} blank TotalCharges values (new customers, tenure=0)")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["churned"] = (df["Churn"] == "Yes").astype(int)
print(f"Overall churn rate: {df['churned'].mean():.1%}")

# ---------------------------------------------------------------
# 1. Churn by contract type
# ---------------------------------------------------------------
contract_churn = df.groupby("Contract")["churned"].mean().sort_values(ascending=False)
print("\nChurn rate by contract type:")
print((contract_churn * 100).round(1))

fig, ax = plt.subplots(figsize=(7, 4.5))
contract_churn.mul(100).plot(kind="bar", ax=ax, color=["#e53e3e", "#dd6b20", "#38a169"])
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Contract Type")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/01_churn_by_contract.png")
plt.close()

# ---------------------------------------------------------------
# 2. Churn by internet service type + tech support
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
service_support = df.groupby(["InternetService", "TechSupport"], observed=True)["churned"].mean().unstack() * 100
service_support.plot(kind="bar", ax=ax)
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Internet Service & Tech Support")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.legend(title="Tech Support")
plt.tight_layout()
plt.savefig("charts/02_churn_by_internet_support.png")
plt.close()

# ---------------------------------------------------------------
# 3. Tenure vs churn
# ---------------------------------------------------------------
df["tenure_bucket"] = pd.cut(
    df["tenure"], bins=[-1, 12, 24, 48, 72],
    labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
)
tenure_churn = df.groupby("tenure_bucket", observed=True)["churned"].mean()

fig, ax = plt.subplots(figsize=(7, 4.5))
tenure_churn.mul(100).plot(kind="line", ax=ax, marker="o", color="#805ad5", linewidth=2)
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Customer Tenure")
plt.tight_layout()
plt.savefig("charts/03_churn_by_tenure.png")
plt.close()

# ---------------------------------------------------------------
# 4. Predictive model
# ---------------------------------------------------------------
model_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
              "PaymentMethod", "InternetService", "TechSupport",
              "PaperlessBilling", "SeniorCitizen"]
features = pd.get_dummies(
    df[model_cols],
    columns=["Contract", "PaymentMethod", "InternetService", "TechSupport", "PaperlessBilling"],
    drop_first=True,
)
target = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=42, stratify=target
)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)

probs = model.predict_proba(X_test_s)[:, 1]
preds = model.predict(X_test_s)
auc = roc_auc_score(y_test, probs)
report = classification_report(y_test, preds)
print(f"\nModel ROC-AUC: {auc:.3f}")
print(report)

fig, ax = plt.subplots(figsize=(6, 6))
RocCurveDisplay.from_predictions(y_test, probs, ax=ax, color="#2b6cb0")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
ax.set_title(f"ROC Curve (AUC = {auc:.2f})")
ax.legend()
plt.tight_layout()
plt.savefig("charts/04_roc_curve.png")
plt.close()

coef_df = pd.DataFrame({
    "feature": X_train_s.columns,
    "coefficient": model.coef_[0]
}).sort_values("coefficient")

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#e53e3e" if c > 0 else "#38a169" for c in coef_df["coefficient"]]
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
ax.set_title("What Drives Churn? (Logistic Regression Coefficients)")
ax.set_xlabel("Coefficient (positive = increases churn risk)")
plt.tight_layout()
plt.savefig("charts/05_feature_importance.png")
plt.close()

# ---------------------------------------------------------------
# 5. Ranked at-risk customer list
# ---------------------------------------------------------------
all_scaled = features.copy()
all_scaled[num_cols] = scaler.transform(features[num_cols])
df["churn_risk_score"] = model.predict_proba(all_scaled)[:, 1]

at_risk = (
    df[df["churned"] == 0]
    .sort_values("churn_risk_score", ascending=False)
    .head(20)
    [["customerID", "tenure", "Contract", "MonthlyCharges", "TechSupport", "churn_risk_score"]]
)
at_risk.to_csv("top_20_at_risk_customers.csv", index=False)
print("\nTop 5 highest-risk active customers:")
print(at_risk.head())

with open("model_results.txt", "w") as f:
    f.write(f"Data cleaning note: {n_missing} blank TotalCharges values found and set to 0 ")
    f.write("(new customers with 0 months tenure, not billed yet)\n\n")
    f.write(f"ROC-AUC: {auc:.3f}\n\n")
    f.write(report)
    f.write("\n\nFeature coefficients:\n")
    f.write(coef_df.to_string(index=False))
