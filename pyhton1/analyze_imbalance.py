import pandas as pd
import matplotlib.pyplot as plt

# Daha önce kaydettiğimiz sonuç dosyası
df = pd.read_csv("experiment_results.csv")

# Dataset isminden normal sayısını çekiyoruz
# exp_dist_500_492.csv → 500
df["Normal_Size"] = df["Dataset"].str.extract(r'exp_dist_(\d+)_').astype(int)

# Fraud sabit olduğu için IR hesaplıyoruz
fraud_count = 492
df["IR"] = df["Normal_Size"] / fraud_count

print(df[["Dataset", "Normal_Size", "IR", "LR Recall", "RF Recall"]])

# --------------------------------------------------
# GRAFİK ÇİZ
# --------------------------------------------------
plt.figure(figsize=(8,5))

plt.plot(df["IR"], df["LR Recall"], marker='o', label="Logistic Regression")
plt.plot(df["IR"], df["RF Recall"], marker='s', label="Random Forest")

plt.xlabel("Imbalance Ratio (Normal / Fraud)")
plt.ylabel("Fraud Recall")
plt.title("Model Performansı vs Class Imbalance")

plt.legend()
plt.grid(True)

plt.show()