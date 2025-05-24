import pandas as pd
import numpy as np
import os

df = pd.read_excel("initial_values\merge_excel.xlsx")
df = df.dropna(how="all")
df = df.dropna(axis=1,how="all")

print('\n')
print(df.head())

imgList = os.path.basename("dataset")
print(imgList)
