
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE


df = pd.read_csv("creditcard_ready.csv")

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

print("Fraud:", len(fraud))
print("Normal:", len(normal))

# -----------------------------
# TEST SET (GERÇEK DÜNYA)
# -----------------------------
fraud_test = fraud.copy()
normal_test = normal.sample(n=1000, random_state=42)

test_df = pd.concat([fraud_test, normal_test]).sample(frac=1, random_state=42)

X_test = test_df.drop("Class", axis=1).values
y_test = test_df["Class"].values

# fraud test dışında kalan normal
normal_train_pool = normal.drop(normal_test.index)

sizes = [500,1000,1500,3000,6000,12000,24000,48000,50000,55000,60000,96000]

results = []

for size in sizes:

    print("\n==============================")
    print(f"SIZE: {size}")
    print("==============================")

    normal_sample = normal_train_pool.sample(n=size, replace=True, random_state=42)

    train_df = pd.concat([normal_sample, fraud]).sample(frac=1, random_state=42)

    X_train = train_df.drop("Class", axis=1).values
    y_train = train_df["Class"].values

    def get_models():
        return {
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "RandomForest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42
            )
        }

    for name, model in get_models().items():

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        results.append([
            name, "No SMOTE", size,

            accuracy_score(y_train, train_pred),
            precision_score(y_train, train_pred, zero_division=0),
            recall_score(y_train, train_pred),
            f1_score(y_train, train_pred),

            accuracy_score(y_test, test_pred),
            precision_score(y_test, test_pred, zero_division=0),
            recall_score(y_test, test_pred),
            f1_score(y_test, test_pred)
        ])

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    for name, model in get_models().items():

        model.fit(X_train_sm, y_train_sm)

        train_pred = model.predict(X_train_sm)
        test_pred = model.predict(X_test)

        results.append([
            name, "SMOTE", size,

            accuracy_score(y_train_sm, train_pred),
            precision_score(y_train_sm, train_pred, zero_division=0),
            recall_score(y_train_sm, train_pred),
            f1_score(y_train_sm, train_pred),

            accuracy_score(y_test, test_pred),
            precision_score(y_test, test_pred, zero_division=0),
            recall_score(y_test, test_pred),
            f1_score(y_test, test_pred)
        ])


columns = [
    "Model", "SMOTE", "Size",
    "Train_Acc", "Train_Prec", "Train_Recall", "Train_F1",
    "Test_Acc", "Test_Prec", "Test_Recall", "Test_F1"
]

results_df = pd.DataFrame(results, columns=columns)

results_df.to_csv("final_experiment_results.csv", index=False)

print("\n✔ DONE")
print(results_df.head(20))