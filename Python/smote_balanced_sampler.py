# smote_balanced_sampler.py
import pandas as pd
import numpy as np
from scipy.stats import zscore
from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard_ready.csv")

fraud = df[df['Class']==1]
normal = df[df['Class']==0]

sizes = [500,1000,1500,3000,6000,12000,24000,48000,50000,55000,60000,96000]

orig_mean = normal.drop('Class',axis=1).mean()
orig_std = normal.drop('Class',axis=1).std()

for target_size in sizes:
    # Normal veri seçimi (dağılım korunuyor)
    normal_z = normal.copy()
    normal_z.iloc[:,:-1] = normal.iloc[:,:-1].apply(zscore)
    bins = np.linspace(-4,4,60)
    sampled_parts = []

    for col in normal_z.columns[:-1]:
        temp = normal_z[[col]].copy()
        temp['bin'] = pd.cut(temp[col], bins=bins)
        grouped = temp.groupby('bin')
        for idx in grouped.groups.values():
            group = normal.loc[idx]
            if len(group) > 5:
                frac = target_size/len(normal)
                sampled = group.sample(frac=frac, replace=(target_size>len(normal)), random_state=42)
                sampled_parts.append(sampled)

    sampled_normal = pd.concat(sampled_parts).drop_duplicates()
    sampled_normal = sampled_normal.sample(n=target_size, replace=(len(sampled_normal)<target_size), random_state=42)
    sampled_normal['Class'] = 0

    temp_df = pd.concat([sampled_normal, fraud])
    X = temp_df.drop('Class',axis=1)
    y = temp_df['Class']

    # SMOTE ile fraud sayısını normal ile eşitle
    smote = SMOTE(sampling_strategy={1: target_size}, random_state=42)
    X_res, y_res = smote.fit_resample(X,y)

    df_balanced = pd.DataFrame(X_res, columns=X.columns)
    df_balanced['Class'] = y_res

    # İstatistik kontrol
    new_normal = df_balanced[df_balanced['Class']==0]
    new_mean = new_normal.drop('Class',axis=1).mean()
    new_std = new_normal.drop('Class',axis=1).std()
    print(f"{target_size} / {target_size} -> Mean farkı: {(orig_mean-new_mean).abs().mean():.4f}, Std farkı: {(orig_std-new_std).abs().mean():.4f}")

    file_name = f"exp_smote_{target_size}_{target_size}.csv"
    df_balanced.to_csv(file_name, index=False)
    print(f"Kaydedildi: {file_name}")