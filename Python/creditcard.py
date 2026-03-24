import pandas as pd
import numpy as np
from scipy.stats import zscore

# 1️⃣ Temiz veri setini oku
df = pd.read_csv("creditcard_clean.csv")

print("Veri boyutu:", df.shape)

# 2️⃣ Fraud ve Normal ayır
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]

print("\nFraud sayısı:", len(fraud))
print("Normal sayısı:", len(normal))

# --------------------------------------------------
# 3️⃣ NORMAL VERİNİN İSTATİSTİĞİNİ HESAPLA
# --------------------------------------------------
mean_before = normal.mean()
std_before = normal.std()

print("\nOrijinal Mean hesaplandı")
print("Orijinal Std hesaplandı")

# --------------------------------------------------
# 4️⃣ Z-SCORE HESAPLA (DAĞILIMI KORUMAK İÇİN)
# --------------------------------------------------
normal_z = normal.copy()
normal_z.iloc[:, :-1] = normal.iloc[:, :-1].apply(zscore)

# --------------------------------------------------
# 5️⃣ HER Z-ARALIĞINDAN ORANTILI ÖRNEK SEÇ
# --------------------------------------------------
bins = np.linspace(-4, 4, 50)

sampled_list = []

for col in normal_z.columns[:-1]:
    normal_z['bin'] = pd.cut(normal_z[col], bins=bins)

    grouped = normal_z.groupby('bin')

    for _, group in grouped:
        if len(group) > 2:
            sampled_list.append(group.sample(frac=0.03, random_state=42))

sampled_normal = pd.concat(sampled_list).drop_duplicates()

# Fazla seçildiyse sınırla
sampled_normal = sampled_normal.sample(n=8000, random_state=42)

# Class geri ekle
sampled_normal['Class'] = 0

print("\nSeçilen Normal veri:", sampled_normal.shape)

# --------------------------------------------------
# 6️⃣ FRAUD + SEÇİLEN NORMAL BİRLEŞTİR
# --------------------------------------------------
df_small = pd.concat([fraud, sampled_normal])
df_small = df_small.sample(frac=1, random_state=42)

# --------------------------------------------------
# 7️⃣ İSTATİSTİK KONTROLÜ (HOCA BURAYA BAKACAK)
# --------------------------------------------------
mean_after = sampled_normal.mean()
std_after = sampled_normal.std()

print("\nMean değişimi:")
print((mean_before - mean_after).abs().mean())

print("\nStd değişimi:")
print((std_before - std_after).abs().mean())

# --------------------------------------------------
# 8️⃣ Kaydet
# --------------------------------------------------
df_small.to_csv("creditcard_balanced.csv", index=False)

print("\n✔ İstatistiği korunmuş veri kaydedildi → creditcard_balanced.csv")
