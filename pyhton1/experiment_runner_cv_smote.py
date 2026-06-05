import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


# -------------------------------
# Değerleri saklamak için listeler
# -------------------------------
dataset_sizes = []

recall_lr_values = []
recall_rf_values = []

accuracy_lr_values = []
precision_lr_values = []
f1_lr_values = []

accuracy_rf_values = []
precision_rf_values = []
f1_rf_values = []

# -------------------------------
# Modeli çalıştırma fonksiyonu
# -------------------------------
def run_model_cv_smote(file_name, n_splits=5):

    print(f"\n========= {file_name} =========")

    df = pd.read_csv(file_name)

    normal_count = sum(df['Class'] == 0)
    fraud_count = sum(df['Class'] == 1)

    IR = normal_count / fraud_count

    print(f"Normal: {normal_count} Fraud: {fraud_count} IR: {IR:.2f}")

    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    lr_recalls = []
    rf_recalls = []

    lr_acc = []
    lr_prec = []
    lr_f1 = []

    rf_acc = []
    rf_prec = []
    rf_f1 = []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        # SMOTE sadece train datasına uygulanır
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Logistic Regression
        lr_model = LogisticRegression(max_iter=2000)
        lr_model.fit(X_train, y_train)

        y_pred_lr = lr_model.predict(X_test)

        lr_recalls.append(recall_score(y_test, y_pred_lr))
        lr_prec.append(precision_score(y_test, y_pred_lr))
        lr_f1.append(f1_score(y_test, y_pred_lr))
        lr_acc.append(accuracy_score(y_test, y_pred_lr))

        # Random Forest
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)

        rf_recalls.append(recall_score(y_test, y_pred_rf))
        rf_prec.append(precision_score(y_test, y_pred_rf))
        rf_f1.append(f1_score(y_test, y_pred_rf))
        rf_acc.append(accuracy_score(y_test, y_pred_rf))

    # -------------------------------
    # Fold ortalamaları
    # -------------------------------

    recall_lr_avg = np.mean(lr_recalls)
    recall_rf_avg = np.mean(rf_recalls)

    acc_lr_avg = np.mean(lr_acc)
    prec_lr_avg = np.mean(lr_prec)
    f1_lr_avg = np.mean(lr_f1)

    acc_rf_avg = np.mean(rf_acc)
    prec_rf_avg = np.mean(rf_prec)
    f1_rf_avg = np.mean(rf_f1)

    print("\n--- Logistic Regression Ortalama ---")
    print(f"Accuracy: {acc_lr_avg:.4f}")
    print(f"Precision: {prec_lr_avg:.4f}")
    print(f"Recall: {recall_lr_avg:.4f}")
    print(f"F1: {f1_lr_avg:.4f}")

    print("\n--- Random Forest Ortalama ---")
    print(f"Accuracy: {acc_rf_avg:.4f}")
    print(f"Precision: {prec_rf_avg:.4f}")
    print(f"Recall: {recall_rf_avg:.4f}")
    print(f"F1: {f1_rf_avg:.4f}")

    # Sonuçları kaydet
    dataset_sizes.append(normal_count)

    recall_lr_values.append(recall_lr_avg)
    recall_rf_values.append(recall_rf_avg)

    accuracy_lr_values.append(acc_lr_avg)
    precision_lr_values.append(prec_lr_avg)
    f1_lr_values.append(f1_lr_avg)

    accuracy_rf_values.append(acc_rf_avg)
    precision_rf_values.append(prec_rf_avg)
    f1_rf_values.append(f1_rf_avg)


# -------------------------------
# SMOTE datasetleri
# -------------------------------
csv_files = [
"exp_smote_500_500.csv","exp_smote_1000_1000.csv","exp_smote_1500_1500.csv",
"exp_smote_3000_3000.csv","exp_smote_6000_6000.csv","exp_smote_12000_12000.csv",
"exp_smote_24000_24000.csv","exp_smote_48000_48000.csv","exp_smote_50000_50000.csv",
"exp_smote_55000_55000.csv","exp_smote_60000_60000.csv","exp_smote_96000_96000.csv"
]

for file in csv_files:
    run_model_cv_smote(file)

# -------------------------------
# Grafik
# -------------------------------
plt.figure(figsize=(8,6))
plt.plot(dataset_sizes, recall_lr_values, marker='o', label="Logistic Regression Recall")
plt.plot(dataset_sizes, recall_rf_values, marker='s', label="Random Forest Recall")
plt.xlabel("Dataset Size (Normal = Fraud)")
plt.ylabel("Fraud Recall")
plt.title("Fraud Recall vs Dataset Size (SMOTE Balanced)")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------
# RAPOR YORUMU
# -------------------------------
print("\n📌 RAPOR YORUMU (SMOTE):")
print("SMOTE ile fraud sınıfı dengelendi. Tüm datasetlerde normal ve fraud sayısı eşit.")
print("Bu sayede model Fraud Recall değerleri yüksek ve stabil kaldı.")
print("Dataset boyutu arttıkça performans artışı gözlenebilir.")