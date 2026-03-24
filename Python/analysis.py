import pandas as pd

# Veri setini oku
df = pd.read_csv("creditcard.csv")

print("Veri boyutu:", df.shape)

print("\nSütunlar:")
print(df.columns)

print("\nİlk 5 Satır:")
print(df.head())

# Eksik veri kontrolü
print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())

# Sınıf dağılımı
print("\nSınıf Dağılımı:")
print(df['Class'].value_counts())

# İstatistiksel özet
print("\nİstatistiksel Özet:")
print(df.describe())
