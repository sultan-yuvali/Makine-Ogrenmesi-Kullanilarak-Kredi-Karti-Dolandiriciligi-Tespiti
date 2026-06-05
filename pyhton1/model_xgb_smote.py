#xgboost smotelu

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("creditcard_ready.csv")

df_fraud = df[df["Class"] == 1]
df_normal = df[df["Class"] == 0]

# -------------------------------
# SENARYOLAR
# -------------------------------
sizes = [500,1000,1500,3000,6000,12000,24000,48000,50000,55000,60000,96000]

for size in sizes:

    print("\n==============================")
    print(f"SMOTE SENARYO: {size}-{size}")
    print("==============================")

    df_normal_sample = df_normal.sample(n=size, replace=True, random_state=42)
    df_scenario = pd.concat([df_normal_sample, df_fraud])
    df_scenario = df_scenario.sample(frac=1, random_state=42)

    X = df_scenario.drop("Class", axis=1).values
    y = df_scenario["Class"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    recalls, precisions, f1s, aucs, accuracies = [], [], [], [], []

    for train_idx, test_idx in skf.split(X, y):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # SMOTE
        smote = SMOTE(sampling_strategy=1.0, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42
        )

        model.fit(X_train_res, y_train_res)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.3).astype(int)

        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))
        accuracies.append(accuracy_score(y_test, y_pred))

    print("Accuracy:", np.mean(accuracies))
    print("Recall:", np.mean(recalls))
    print("Precision:", np.mean(precisions))
    print("F1:", np.mean(f1s))
    print("AUC:", np.mean(aucs))


print("\nFINAL SMOTE MODEL (96000)")

df_normal_sample = df_normal.sample(n=96000, replace=True, random_state=42)
df_final = pd.concat([df_normal_sample, df_fraud])
df_final = df_final.sample(frac=1, random_state=42)

X = df_final.drop("Class", axis=1).values
y = df_final["Class"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_test = []
all_y_pred = []

for train_idx, test_idx in skf.split(X, y):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SMOTE
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.3).astype(int)

    all_y_test.extend(y_test)
    all_y_pred.extend(y_pred)

cm = confusion_matrix(all_y_test, all_y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SMOTE XGBoost Confusion Matrix (96000)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()