# ==============================
# 1. KÜTÜPHANELER
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 2. VERİYİ YÜKLE
# ==============================
df = pd.read_csv("creditcard.csv")  # kendi dosyan

# ==============================
# 3. X - y AYIR
# ==============================
X = df.drop("Class", axis=1)
y = df["Class"]

# ==============================
# 4. SMOTE
# ==============================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("SMOTE sonrası sınıf dağılımı:")
print(y_res.value_counts())

# ==============================
# 5. TRAIN - TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=42
)

# ==============================
# 6. KORELASYON ANALİZİ
# ==============================
corr = pd.concat([X_train, y_train], axis=1).corr()["Class"].sort_values(ascending=False)

print("\nEn önemli feature'lar (korelasyona göre):")
print(corr.head(10))

# Heatmap (isteğe bağlı)
plt.figure(figsize=(10,6))
sns.heatmap(pd.concat([X_train, y_train], axis=1).corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ==============================
# 7. PCA
# ==============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nPCA Variance:")
print(pca.explained_variance_ratio_)

# ==============================
# 8. RANDOM FOREST MODEL
# ==============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_pca, y_train)

y_pred = rf.predict(X_test_pca)

# ==============================
# 9. PERFORMANCE
# ==============================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# 10. CONFUSION MATRIX 🔥
# ==============================
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ==============================
# 11. FEATURE IMPORTANCE 🔥
# ==============================
# (PCA kullandıysan direkt feature adı yok, o yüzden PCA öncesi modelle bakıyoruz)

rf2 = RandomForestClassifier(n_estimators=100, random_state=42)
rf2.fit(X_train, y_train)

importances = rf2.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nEn kritik feature'lar (Random Forest):")
print(feature_importance.head(10))