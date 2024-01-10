import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np


input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")

df = pd.read_csv(input_file_name)
df.drop(['Time Elapsed (s)',' Total Memory Usage (%)'], inplace = True, axis = 1)

corr = df.corr()
f, ax = plt.subplots(figsize=(28, 20))
heatmap = sn.heatmap(corr,vmin=-1.0,vmax=1.0, annot=True,)

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)

plt.show()

instructions_data = df['Instructions']
cycles_data = df['Cycles']
branches_data = df['Branches']

# Calculate the mean and standard deviation for each dataset
mean_instructions = np.mean(instructions_data)
std_dev_instructions = np.std(instructions_data)

mean_cycles = np.mean(cycles_data)
std_dev_cycles = np.std(cycles_data)

mean_branches = np.mean(branches_data)
std_dev_branches = np.std(branches_data)

# Create the x-axis values
x = np.linspace(0, max(mean_instructions, mean_cycles, mean_branches) + 3 * max(std_dev_instructions, std_dev_cycles, std_dev_branches), 100)

# Calculate the Gaussian distributions based on the data's mean and standard deviation for each metric
pdf_instructions = 1 / (std_dev_instructions * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_instructions) / std_dev_instructions) ** 2)
pdf_cycles = 1 / (std_dev_cycles * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_cycles) / std_dev_cycles) ** 2)
pdf_branches = 1 / (std_dev_branches * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean_branches) / std_dev_branches) ** 2)

# Plot the Gaussian curves for each metric
plt.plot(x, pdf_instructions, color='blue', linestyle='-', linewidth=2, label='Instructions')
plt.plot(x, pdf_cycles, color='red', linestyle='-', linewidth=2, label='Cycles')
plt.plot(x, pdf_branches, color='green', linestyle='-', linewidth=2, label='Branches')

plt.title('Gaussian Distributions for Instructions, Cycles, and Branches')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()
print("Instructions: Mean =", mean_instructions, "Standard Deviation =", std_dev_instructions)
print("Cycles: Mean =", mean_cycles, "Standard Deviation =", std_dev_cycles)
print("Branches: Mean =", mean_branches, "Standard Deviation =", std_dev_branches)


