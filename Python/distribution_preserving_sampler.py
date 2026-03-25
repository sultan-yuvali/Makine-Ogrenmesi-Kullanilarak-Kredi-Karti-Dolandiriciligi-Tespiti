import pandas as pd
import numpy as np
from scipy.stats import zscore

# --------------------------------------------------
# 1️⃣ VERİYİ OKU
# --------------------------------------------------
df = pd.read_csv("creditcard_ready.csv")

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

print("Fraud:", len(fraud))
print("Normal:", len(normal))

# Hocanın istediği artış senaryoları
sizes = [500, 1000, 1500, 3000, 6000, 12000, 24000, 48000, 50000, 55000, 60000, 96000]

# --------------------------------------------------
# 2️⃣ ORİJİNAL DAĞILIM (REFERANS)
# --------------------------------------------------
orig_mean = normal.drop("Class", axis=1).mean()
orig_std  = normal.drop("Class", axis=1).std()

# --------------------------------------------------
# 3️⃣ HER SENARYO İÇİN DAĞILIM KORUYARAK ÖRNEK SEÇ
# --------------------------------------------------
for target_size in sizes:

    print(f"\n=== {target_size} NORMAL SEÇİLİYOR ===")

    # Z-score uzayına geç (dağılımı görmek için)
    normal_z = normal.copy()
    normal_z.iloc[:, :-1] = normal.iloc[:, :-1].apply(zscore)

    bins = np.linspace(-4, 4, 60)
    sampled_parts = []

    # ❗ bin sütununu ana veriye EKLEMEDEN çalışıyoruz
    for col in normal_z.columns[:-1]:

        temp = normal_z[[col]].copy()          # geçici dataframe
        temp["bin"] = pd.cut(temp[col], bins=bins)

        grouped = temp.groupby("bin")

        for idx in grouped.groups.values():
            group = normal.loc[idx]

            if len(group) > 5:
                frac = target_size / len(normal)

                # Eğer hedef sayı elimizdeki veriden büyükse → UPSAMPLING
                if target_size > len(normal):
                    sampled = group.sample(frac=frac, replace=True, random_state=42)
                else:
                    sampled = group.sample(frac=frac, replace=False, random_state=42)

                sampled_parts.append(sampled)
    sampled_normal = pd.concat(sampled_parts).drop_duplicates()

    # Tam hedef sayıya indir
    sampled_normal = sampled_normal.sample(
        n=target_size,
        replace=(len(sampled_normal) < target_size),
        random_state=42
    )

    # --------------------------------------------------
    # 4️⃣ FRAUD İLE BİRLEŞTİR (fraud sabit kalıyor!)
    # --------------------------------------------------
    df_exp = pd.concat([sampled_normal, fraud])
    df_exp = df_exp.sample(frac=1, random_state=42)

    # --------------------------------------------------
    # 5️⃣ İSTATİSTİK KONTROL (hocanın istediği kanıt)
    # --------------------------------------------------
    new_mean = sampled_normal.drop("Class", axis=1).mean()
    new_std  = sampled_normal.drop("Class", axis=1).std()

    print("Mean farkı:", (orig_mean - new_mean).abs().mean())
    print("Std farkı :", (orig_std - new_std).abs().mean())

    # --------------------------------------------------
    # 6️⃣ KAYDET
    # --------------------------------------------------
    file_name = f"exp_dist_{target_size}_492.csv"
    df_exp.to_csv(file_name, index=False)

    print("Kaydedildi:", file_name)

print("\n✔ TÜM SENARYOLAR OLUŞTURULDU.")