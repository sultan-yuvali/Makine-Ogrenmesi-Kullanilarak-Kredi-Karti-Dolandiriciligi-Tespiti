import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1️⃣ Dengelenmiş veriyi oku
df = pd.read_csv("creditcard_balanced.csv")

print("Normalizasyon öncesi Amount:")
print(df[['Amount']].head())

# 2️⃣ Amount normalize edilir (0-1 arası)
scaler = MinMaxScaler()
df[['Amount']] = scaler.fit_transform(df[['Amount']])

print("\nNormalizasyon sonrası Amount:")
print(df[['Amount']].head())

# 3️⃣ Kaydet
df.to_csv("creditcard_ready.csv", index=False)

print("\n✔ Model için hazır veri → creditcard_ready.csv")
