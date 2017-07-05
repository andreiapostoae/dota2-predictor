import csv
import pandas as pd

df = pd.read_csv("part3.csv")

cols = df.columns.tolist()

cols = [cols[0]] + [cols[-1]] + cols[1:-1]

df = df[cols]

df.to_csv("part3_repaired.csv", sep=',', index=False)