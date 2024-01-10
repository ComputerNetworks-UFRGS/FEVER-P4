import pandas as pd


input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file_name)

# Remove dots (periods) from the values in the first three columns
for col in df.columns[1:5]:
    df[col] = df[col].astype(str).str.replace('.', '')
df['Task Clock (msec)'] = df['Task Clock (msec)'].apply(lambda x: x.replace('.', '', 1) if x.count('.') == 2 else x)

df['Context Switches'] = df['Context Switches'].astype(str).str.replace('.', '')
# Save the modified data to a new CSV file
df.to_csv(input_file_name, index=False)

print(f"Modified data saved to {input_file_name}")

