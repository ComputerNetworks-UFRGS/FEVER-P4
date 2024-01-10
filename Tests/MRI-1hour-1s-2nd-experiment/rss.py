from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyod.models.lof import LOF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.ocsvm import OCSVM

def calculate_f1_score_lof(X_train, X_test):
    clf = LOF(n_neighbors=50, contamination=contamination_factor)
    clf.fit(X_train)
    pred_test = clf.predict(X_test)
    
    true_positive_rate = np.sum(pred_test == 1) / len(pred_test)
    false_positive_rate = np.sum(pred_test == 0) / len(pred_test)
    precision = np.sum(pred_test == 1) / np.sum(pred_test)
    f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)
    
    return f1

def calculate_f1_score_lof_true(X_train, X_test):
    clf = LOF(n_neighbors=50, contamination=contamination_factor)
    clf.fit(X_train)
    pred_test = clf.predict(X_test)
    
    true_positive_rate = np.sum(pred_test == 0) / len(pred_test)
    false_positive_rate = np.sum(pred_test == 1) / len(pred_test)
    precision = np.sum(pred_test == 1) / np.sum(pred_test)
    f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)
    
    return f1


def calculate_f1_score_vm(X_train, X_test):
    clf = OCSVM(kernel='rbf',gamma=0.0001, nu=0.3, contamination=contamination_factor)
    clf.fit(X_train)
    pred_test = clf.predict(X_test)
    
    true_positive_rate = np.sum(pred_test == 1) / len(pred_test)
    false_positive_rate = np.sum(pred_test == 0) / len(pred_test)
    precision = np.sum(pred_test == 1) / np.sum(pred_test)
    f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)
    
    return f1
    
def calculate_f1_score_vm_true(X_train, X_test):
    clf = OCSVM(kernel='rbf',gamma=0.0001, nu=0.3, contamination=contamination_factor)
    clf.fit(X_train)
    pred_test = clf.predict(X_test)
    
    true_positive_rate = np.sum(pred_test == 0) / len(pred_test)
    false_positive_rate = np.sum(pred_test == 1) / len(pred_test)
    precision = np.sum(pred_test == 1) / np.sum(pred_test)
    f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)
    
    return f1

def plot_f1_scores(features, f1_scores_lof, f1_scores_vm, title):
    plt.figure(figsize=(10, 6))
    X_axis = np.arange(len(features))
    plt.bar(X_axis - 0.2, f1_scores_lof, 0.4, label='Anomalous', alpha=0.75, color='#404040', edgecolor='black')
    plt.bar(X_axis + 0.2, f1_scores_vm, 0.4, label='Normal', alpha=0.75, color='gray', edgecolor='black')
    plt.xticks(X_axis, features)
    plt.xlabel('Feature')
    plt.ylabel('F1-Score')
    plt.title(title)
    
    # Add text labels for F1-Score values
    for i, (f1_lof, f1_vm) in enumerate(zip(f1_scores_lof, f1_scores_vm)):
        plt.text(X_axis[i] - 0.2, f1_lof + 0.01, f'{f1_lof:.2f}', ha='center', va='bottom')
        plt.text(X_axis[i] + 0.2, f1_vm + 0.01, f'{f1_vm:.2f}', ha='center', va='bottom')

    plt.legend()
    plt.show()

def plot_f1_scores_modified(features, f1_scores_lof, f1_scores_vm, title):
    plt.figure(figsize=(10, 6))
    X_axis = np.arange(len(features))
    plt.bar(X_axis - 0.2, f1_scores_lof, 0.4, label='Modified', alpha=0.75, color='#404040', edgecolor='black')
    plt.bar(X_axis + 0.2, f1_scores_vm, 0.4, label='Normal', alpha=0.75, color='gray', edgecolor='black')
    plt.xticks(X_axis, features)
    plt.xlabel('Feature')
    plt.ylabel('F1-Score')
    plt.title(title)
    
    # Add text labels for F1-Score values
    for i, (f1_lof, f1_vm) in enumerate(zip(f1_scores_lof, f1_scores_vm)):
        plt.text(X_axis[i] - 0.2, f1_lof + 0.01, f'{f1_lof:.2f}', ha='center', va='bottom')
        plt.text(X_axis[i] + 0.2, f1_vm + 0.01, f'{f1_vm:.2f}', ha='center', va='bottom')

    plt.legend()
    plt.show()

input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")
input_file_name2 = input("Enter the anomalous input CSV file name (e.g., name.csv): ")
input_file_name3 = input("Enter the modified input CSV file name (e.g., name.csv): ")

df_normal = pd.read_csv(input_file_name)
df_anomalous = pd.read_csv(input_file_name2)
df_modified = pd.read_csv(input_file_name3)

df_modified.replace('<not', 0, inplace=True)

df_normal.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)'], inplace=True, axis=1)
df_anomalous.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)'], inplace=True, axis=1)
df_modified.drop(['Time Elapsed (s)', ' RSS (%)', ' Virtual Memory Size', ' Total Memory Usage (%)'], inplace=True, axis=1)

# Split dataset in training and testing
X_train, X_test = train_test_split(df_normal, test_size=0.10, random_state=42, shuffle=True)

# Feature scaling
scaler = StandardScaler().fit(X_train)

# Transform training set
X_train = scaler.transform(X_train)

# Transform test set
X_test = scaler.transform(X_test)


# Transform anomalous set
X_test_anomalous = scaler.transform(df_anomalous)

# Transform modified set
X_test_modified = scaler.transform(df_modified)

# Replace NaN values with the mean (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_test_anomalous = imputer.transform(X_test_anomalous)
X_test_modified = imputer.transform(X_test_modified)

contamination_factor = 0.05

# ... (previous code remains unchanged)

features = ['Instructions', ['Instructions', ' RSS (KB)'], ' RSS (KB)']

f1_scores_lof = []
f1_scores_anomalous_lof = []
f1_scores_modified_lof = []

f1_scores_vm = []
f1_scores_anomalous_vm = []
f1_scores_modified_vm = []

# Remove unnecessary reversal
# features.reverse()

for i in range(len(features)):
    current_features = features[i]
    
    # Selecting only the current feature(s)
    if isinstance(current_features, list):
        X_train_current_features = pd.DataFrame(X_train, columns=df_normal.columns)[current_features]
        X_test_current_features = pd.DataFrame(X_test, columns=df_normal.columns)[current_features]
        X_test_anomalous_current_features = pd.DataFrame(X_test_anomalous, columns=df_normal.columns)[current_features]
        X_test_modified_current_features = pd.DataFrame(X_test_modified, columns=df_normal.columns)[current_features]
    else:
        X_train_current_features = pd.DataFrame(X_train, columns=df_normal.columns)[[current_features]]
        X_test_current_features = pd.DataFrame(X_test, columns=df_normal.columns)[[current_features]]
        X_test_anomalous_current_features = pd.DataFrame(X_test_anomalous, columns=df_normal.columns)[[current_features]]
        X_test_modified_current_features = pd.DataFrame(X_test_modified, columns=df_normal.columns)[[current_features]]

    # Calculating F1-Score
    f1 = calculate_f1_score_lof_true(X_train_current_features, X_test_current_features)
    f1_scores_lof.append(f1)
    f1 = calculate_f1_score_vm_true(X_train_current_features, X_test_current_features)
    f1_scores_vm.append(f1)

    f1 = calculate_f1_score_lof(X_train_current_features, X_test_anomalous_current_features)
    f1_scores_anomalous_lof.append(f1)
    f1 = calculate_f1_score_vm(X_train_current_features, X_test_anomalous_current_features)
    f1_scores_anomalous_vm.append(f1)

    f1 = calculate_f1_score_lof(X_train_current_features, X_test_modified_current_features)
    f1_scores_modified_lof.append(f1)
    f1 = calculate_f1_score_vm(X_train_current_features, X_test_modified_current_features)
    f1_scores_modified_vm.append(f1)

# Plotting F1-Scores
plot_f1_scores(features, f1_scores_anomalous_lof, f1_scores_lof, 'LOF & OCSVM F1-Scores (Anomalous)')
plot_f1_scores(features, f1_scores_anomalous_vm, f1_scores_vm, 'OCSVM F1-Scores (Anomalous)')
plot_f1_scores_modified(features, f1_scores_modified_vm, f1_scores_vm, 'LOF & OCSVM F1-Scores (Modified)')
plot_f1_scores_modified(features, f1_scores_modified_vm, f1_scores_vm, 'OCSVM F1-Scores (Modified)')

