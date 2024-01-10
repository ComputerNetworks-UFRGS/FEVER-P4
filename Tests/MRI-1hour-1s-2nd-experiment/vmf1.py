from sklearn.model_selection import train_test_split
from scipy import stats
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_f1_score(X_train, X_test):
    clf = OCSVM(kernel='rbf',gamma=0.0001, nu=0.3, contamination=contamination_factor)
    clf.fit(X_train)
    pred_test = clf.predict(X_test)
    
    true_positive_rate = np.sum(pred_test == 1) / len(pred_test)
    false_positive_rate = np.sum(pred_test == 0) / len(pred_test)
    precision = np.sum(pred_test == 1) / np.sum(pred_test)
    f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)
    
    return f1

def plot_f1_scores(features, f1_scores, title):
    plt.figure(figsize=(10, 6))
    plt.bar(features, f1_scores, color='skyblue')
    plt.xlabel('Feature')
    plt.ylabel('F1-Score')
    plt.title(title)
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

features = ['Cycles', 'Instructions', 'Branches', 'Branch Misses', 'Task Clock (msec)',
            'Context Switches', 'CPU Migrations', 'Page Faults', 'RSS (KB)']
f1_scores = []
f1_scores_anomalous = []
f1_scores_modified = []

for feature in features:
    # Selecting only the current feature
    X_train_single_feature = X_train[:, [features.index(feature)]]
    X_test_single_feature = X_test[:, [features.index(feature)]]

    # Calculating F1-Score
    f1 = calculate_f1_score(X_train_single_feature, X_test_single_feature)
    f1_scores.append(f1)


for feature in features:
    # Selecting only the current feature
    X_train_single_feature = X_train[:, [features.index(feature)]]
    X_test_anomalous_single_feature = X_test_anomalous[:, [features.index(feature)]]

    # Calculating F1-Score
    f1 = calculate_f1_score(X_train_single_feature, X_test_anomalous_single_feature)
    f1_scores_anomalous.append(f1)
    
for feature in features:
    # Selecting only the current feature
    X_train_single_feature = X_train[:, [features.index(feature)]]
    X_test_modified_single_feature = X_test_modified[:, [features.index(feature)]]

    # Calculating F1-Score
    f1 = calculate_f1_score(X_train_single_feature, X_test_modified_single_feature)
    f1_scores_modified.append(f1)
    
# Plotting F1-Scores
plot_f1_scores(features, f1_scores, 'OCSVM F1-Scores for Each Feature')
plot_f1_scores(features, f1_scores_anomalous, 'OCSVM F1-Scores for Each Feature (Anomalous)')
plot_f1_scores(features, f1_scores_modified, 'OCSVM F1-Scores for Each Feature(Modified)')

