import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")

df = pd.read_csv(input_file_name)
df.drop(['Time Elapsed (s)',' Total Memory Usage (%)'], inplace = True, axis = 1)

corr = df.corr()
f, ax = plt.subplots(figsize=(28, 20))
heatmap = sn.heatmap(corr,vmin=-1.0,vmax=1.0, annot=True)
plt.show()
