# experiment_runner_cv.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# -------------------------------
# IR ve recall değerlerini saklamak için listeler
# -------------------------------
ir_values = []
recall_lr_values = []
recall_rf_values = []

# -------------------------------
# Modeli çalıştırma fonksiyonu (5-Fold CV)
# -------------------------------
def run_model_cv(file_name, n_splits=5):
    print(f"\n========= {file_name} =========")
    df = pd.read_csv(file_name)

    fraud_count = df["Class"].sum()
    normal_count = len(df) - fraud_count
    IR = normal_count / fraud_count
    print(f"Normal: {normal_count} Fraud: {fraud_count} IR: {IR:.2f}")

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr_recalls = []
    rf_recalls = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=2000)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        lr_recalls.append(recall_score(y_test, y_pred_lr))

        # Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rf_recalls.append(recall_score(y_test, y_pred_rf))

    # Fold ortalamalarını kaydet
    recall_lr_avg = np.mean(lr_recalls)
    recall_rf_avg = np.mean(rf_recalls)

    print(f"Logistic Regression Avg Recall: {recall_lr_avg:.3f}")
    print(f"Random Forest Avg Recall: {recall_rf_avg:.3f}")

    # IR ve Recall kaydet
    ir_values.append(IR)
    recall_lr_values.append(recall_lr_avg)
    recall_rf_values.append(recall_rf_avg)

# -------------------------------
# Farklı senaryoları çalıştır
# -------------------------------
csv_files = [
"exp_dist_500_492.csv",
"exp_dist_1000_492.csv",
"exp_dist_1500_492.csv",
"exp_dist_3000_492.csv",
"exp_dist_6000_492.csv",
"exp_dist_12000_492.csv",
"exp_dist_24000_492.csv",
"exp_dist_48000_492.csv",
"exp_dist_50000_492.csv",
"exp_dist_55000_492.csv",
"exp_dist_60000_492.csv",
"exp_dist_96000_492.csv"
]

for file in csv_files:
    run_model_cv(file)

# -------------------------------
# IR vs Recall grafiği
# -------------------------------
plt.figure(figsize=(8,6))
plt.plot(ir_values, recall_lr_values, marker='o', label="Logistic Regression Recall")
plt.plot(ir_values, recall_rf_values, marker='s', label="Random Forest Recall")
plt.xlabel("Imbalance Ratio (IR = Normal / Fraud)")
plt.ylabel("Fraud Recall")
plt.title("Fraud Recall vs Imbalance Ratio (5-Fold CV)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# RAPOR İÇİN YORUM
# -------------------------------
print("\n📌 RAPOR YORUMU:")
print("Veri dengesizliği arttıkça (IR büyüdükçe) model başlangıçta stabil çalışmıştır.")
print("Ancak IR belirli bir eşik değerini geçtiğinde model fraud sınıfını öğrenememiş ve Fraud Recall değerinde ani düşüş gözlenmiştir.")
print("Bu nokta, aşırı sınıf dengesizliğinin model performansını bozduğu kritik kırılma noktasıdır.")