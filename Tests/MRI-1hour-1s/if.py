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

input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")
input_file_name2 = input("Enter the anomalous input CSV file name (e.g., name.csv): ")
input_file_name3 = input("Enter the modified input CSV file name (e.g., name.csv): ")

df_normal = pd.read_csv(input_file_name)
df_anomalous = pd.read_csv(input_file_name2)
df_modified = pd.read_csv(input_file_name3)

df_normal.drop(['Time Elapsed (s)','Cycles','Instructions','Branches','Branch Misses','Task Clock (msec)','Context Switches','CPU Migrations','Page Faults','Total CPU Usage (%)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)
df_anomalous.drop(['Time Elapsed (s)','Cycles','Instructions','Branches','Branch Misses','Task Clock (msec)','Context Switches','CPU Migrations','Page Faults','Total CPU Usage (%)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)
df_modified.drop(['Time Elapsed (s)','Cycles','Instructions','Branches','Branch Misses','Task Clock (msec)','Context Switches','CPU Migrations','Page Faults','Total CPU Usage (%)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)


# Split dataset in training and testing
X_train, X_test = train_test_split(df_normal, test_size=0.10, random_state=42,shuffle=True)
X_train.shape

X_train=X_train[(np.abs(stats.zscore(X_train)) < 3).all(axis=1)]

#Feature scaling
scaler= StandardScaler().fit(X_train)

# Transform training set
X_train = scaler.transform(X_train)

# Transform test set
X_test = scaler.transform(X_test)

# Transform anomalous set
X_test_anomalous = scaler.transform(df_anomalous)

# Transform modified set
X_test_modified = scaler.transform(df_modified)

contamination_factor=0.05

clf = IForest()

# Model training
clf.fit(X_train)

# Model evaluation with good behaviour
pred_test=clf.predict(X_test)
unique_elements, counts_elements = np.unique(pred_test, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

# Model evaluation with good behaviour
pred_train=clf.predict(X_train)
unique_elements, counts_elements = np.unique(pred_train, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

true_positive_rate = np.sum(pred_train == 1) / len(pred_train)
false_positive_rate = np.sum(pred_train == 0) / len(pred_train)
precision = np.sum(pred_train == 1) / np.sum(pred_train)
f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)

print("Anomaly Detection Metrics:")
print("True Positive Rate (Recall): {:.4f}".format(true_positive_rate))
print("False Positive Rate: {:.4f}".format(false_positive_rate))
print("Precision: {:.4f}".format(precision))
print("F1-score: {:.4f}".format(f1))

# Model evaluation with anomalous behaviour
pred_anomalous=clf.predict(X_test_anomalous)
unique_elements, counts_elements = np.unique(pred_anomalous, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

true_positive_rate = np.sum(pred_anomalous == 1) / len(pred_anomalous)
false_positive_rate = np.sum(pred_anomalous == 0) / len(pred_anomalous)
precision = np.sum(pred_anomalous == 1) / np.sum(pred_anomalous)
f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)

print("Anomaly Detection Metrics:")
print("True Positive Rate (Recall): {:.4f}".format(true_positive_rate))
print("False Positive Rate: {:.4f}".format(false_positive_rate))
print("Precision: {:.4f}".format(precision))
print("F1-score: {:.4f}".format(f1))

# Model evaluation with modified behaviour
pred_modified=clf.predict(X_test_modified)
unique_elements, counts_elements = np.unique(pred_modified, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

true_positive_rate = np.sum(pred_modified == 1) / len(pred_modified)
false_positive_rate = np.sum(pred_modified == 0) / len(pred_modified)
precision = np.sum(pred_modified == 1) / np.sum(pred_modified)
f1 = 2 * (precision * true_positive_rate) / (precision + true_positive_rate)

print("Anomaly Detection Metrics:")
print("True Positive Rate (Recall): {:.4f}".format(true_positive_rate))
print("False Positive Rate: {:.4f}".format(false_positive_rate))
print("Precision: {:.4f}".format(precision))
print("F1-score: {:.4f}".format(f1))
