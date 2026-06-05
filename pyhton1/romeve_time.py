import pandas as pd

# 1️⃣ Ham veri setini oku
df = pd.read_csv("creditcard.csv")

print("Orijinal veri boyutu:", df.shape)
print("Sütunlar:", df.columns)

# 2️⃣ Time sütununu kaldır (Noise olduğu için)
df.drop(columns=["Time"], inplace=True, errors="ignore")

print("\nTime kaldırıldıktan sonra sütunlar:")
print(df.columns)

# 3️⃣ Temiz veriyi kaydet
df.to_csv("creditcard_clean.csv", index=False)

print("\n✔ Temiz veri oluşturuldu → creditcard_clean.csv")
print("Yeni veri boyutu:", df.shape)
