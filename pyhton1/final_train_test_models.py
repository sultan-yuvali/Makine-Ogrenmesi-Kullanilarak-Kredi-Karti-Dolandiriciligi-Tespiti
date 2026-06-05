import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard_ready.csv")

df_fraud = df[df["Class"] == 1]
df_normal = df[df["Class"] == 0]

print("Fraud:", len(df_fraud))
print("Normal:", len(df_normal))

print("\n==============================")
print(" SMOTE YOK (IMBALANCED)")
print("==============================")


df_normal_sample = df_normal.sample(n=1000, random_state=42)

df_imbalanced = pd.concat([df_normal_sample, df_fraud])
df_imbalanced = df_imbalanced.sample(frac=1, random_state=42)

X = df_imbalanced.drop("Class", axis=1)
y = df_imbalanced["Class"]

#  TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="auc",
        random_state=42
    )
}

# -------------------------------
# TRAIN + TEST
# -------------------------------
for name, model in models.items():

    print(f"\n--- {name} ---")

    model.fit(X_train, y_train)

    # TRAIN
    y_pred_train = model.predict(X_train)

    # TEST
    y_pred_test = model.predict(X_test)

    print("TRAIN:")
    print("Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Recall:", recall_score(y_train, y_pred_train))
    print("Precision:", precision_score(y_train, y_pred_train))
    print("F1:", f1_score(y_train, y_pred_train))

    print("TEST:")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Recall:", recall_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test))
    print("F1:", f1_score(y_test, y_pred_test))



print("\n==============================")
print(" SMOTE VAR (BALANCED)")
print("==============================")

#  1000 normal + 492 fraud → sonra SMOTE ile 1000-1000 olacak

df_normal_sample = df_normal.sample(n=1000, random_state=42)
df_balanced = pd.concat([df_normal_sample, df_fraud])
df_balanced = df_balanced.sample(frac=1, random_state=42)

X = df_balanced.drop("Class", axis=1)
y = df_balanced["Class"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#  SADECE TRAIN'E SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

for name, model in models.items():

    print(f"\n--- {name} ---")

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("TRAIN:")
    print("Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Recall:", recall_score(y_train, y_pred_train))
    print("Precision:", precision_score(y_train, y_pred_train))
    print("F1:", f1_score(y_train, y_pred_train))

    print("TEST:")
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Recall:", recall_score(y_test, y_pred_test))
    print("Precision:", precision_score(y_test, y_pred_test))
    print("F1:", f1_score(y_test, y_pred_test))