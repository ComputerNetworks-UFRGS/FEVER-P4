import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_avg_feature_values(df1, df2, df3, df_baseline, title):
    # Calculate average values for each feature
    avg_baseline = df_baseline.mean()
    avg_f1 = df1.mean()
    avg_f2 = df2.mean()
    avg_f3 = df3.mean()

    # Calculate the ratios as percentages
    ratio_f1 = (avg_f1 / avg_baseline) * 100
    ratio_f2 = (avg_f2 / avg_baseline) * 100
    ratio_f3 = (avg_f3 / avg_baseline) * 100

    # Set up the plot
    plt.figure(figsize=(12, 6))
    X_axis = np.arange(len(avg_f1))
    bar_width = 0.25  # Adjust this value to space the bars more or less

    # Plot the average values for each feature
    plt.bar(X_axis - bar_width, ratio_f1, bar_width, label='MRI', alpha=0.75, color='#404040', edgecolor='black')
    plt.bar(X_axis, ratio_f2, bar_width, label='Link Monitor', alpha=0.75, color='gray', edgecolor='black')
    plt.bar(X_axis + bar_width, ratio_f3, bar_width, label='ECN', alpha=0.75, color='lightgray', edgecolor='black')

    plt.xticks(X_axis, avg_f1.index, rotation=45, ha='right')  # Rotate the x-axis labels
    plt.xlabel('Feature')
    plt.ylabel('(%) of Baseline')
    plt.title(title)
    plt.legend()
    plt.show()



features = ['Cycles', 'Instructions', 'Branches', 'Branch Misses', 'Task Clock (msec)',
            'Context Switches', 'CPU Migrations', 'Page Faults', ' RSS (KB)']
            
input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")
input_file_name2 = input("Enter the anomalous input CSV file name (e.g., name.csv): ")
input_file_name3 = input("Enter the modified input CSV file name (e.g., name.csv): ")
input_file_name4 = input("Enter the baseline input CSV file name (e.g., name.csv): ")

df_mri = pd.read_csv(input_file_name)
df_link_monitor = pd.read_csv(input_file_name2)
df_load_balance = pd.read_csv(input_file_name3)
df_baseline = pd.read_csv(input_file_name4)

df_mri.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)', 'Total CPU Usage (%)'], inplace=True, axis=1)
df_link_monitor.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)', 'Total CPU Usage (%)'], inplace=True, axis=1)
df_load_balance.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)', 'Total CPU Usage (%)'], inplace=True, axis=1)
df_baseline.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)', 'Total CPU Usage (%)'], inplace=True, axis=1)

plot_avg_feature_values(df_mri, df_link_monitor, df_load_balance, df_baseline, 'Average Plot')
