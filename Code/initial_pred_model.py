import pandas as pd
import numpy as np
import os

df = pd.read_excel("initial_values\merge_excel.xlsx")
df = df.dropna(how="any")
df = df.dropna(axis=1,how="all")
df = df.drop(columns=['First Name', 'Last Name', 'User Name', 'Created On', 'Creator User ID', 'Date Of Birth', 'Gender', 'Last Modified On', 'Last Period Date', 'Socio-economic status', 'Patient Document Id','Patient ID'])

print('\n')
print(df)


imgList = os.listdir("dataset")
imgListBase = []

for i in imgList:

    imgListBase.append(os.path.splitext(i)[0])

for col in df.columns:

    if df[col].dtype == np.float64:
        df[col] = df[col].astype(int)
    
print('\n')
print(df.dtypes)
