import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")

df = pd.read_csv(input_file_name)
df.drop(['Time Elapsed (s)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)

df = df.apply(pd.to_numeric, errors='coerce')

corr = df.corr()
f, ax = plt.subplots(figsize=(28, 20))
heatmap = sn.heatmap(corr,vmin=-1.0,vmax=1.0, annot=True,)

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)

plt.show()

input_file_name2 = input("Enter the second input CSV file name (e.g., name.csv): ")


df2 = pd.read_csv(input_file_name2)

df2 = df2.apply(pd.to_numeric, errors='coerce')

#Reset index to plot
df.reset_index(inplace=True,drop=True)
df2.reset_index(inplace=True,drop=True)

for feat_hist in df.columns:

    data_hist_0 = df[feat_hist]
    data_hist_0 = data_hist_0[data_hist_0.between(data_hist_0.quantile(.01), data_hist_0.quantile(.99))]

    data_hist_1 = df2[feat_hist]
    data_hist_1 = data_hist_1[data_hist_1.between(data_hist_1.quantile(.01), data_hist_1.quantile(.99))]
    

    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(data_hist_0,label=feat_hist+"_swnormal",color="blue")
    plt.plot(data_hist_1,label=feat_hist+"_swmodificado",color="green")


    plt.legend()
    plt.show()




