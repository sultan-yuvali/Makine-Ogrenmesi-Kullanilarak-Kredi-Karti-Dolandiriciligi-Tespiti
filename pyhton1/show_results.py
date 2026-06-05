import pandas as pd
from tabulate import tabulate

df = pd.read_csv("final_experiment_results.csv")
for size in sorted(df["Size"].unique()):
    print("\n====================")
    print("SIZE:", size)
    print("====================")

    print(tabulate(df[df["Size"] == size],
                   headers='keys',
                   tablefmt='grid',
                   showindex=False))

