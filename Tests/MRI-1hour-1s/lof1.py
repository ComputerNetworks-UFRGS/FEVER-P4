from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from pyod.models.lof import LOF
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns



input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")
input_file_name2 = input("Enter the anomalous input CSV file name (e.g., name.csv): ")
input_file_name3 = input("Enter the modified input CSV file name (e.g., name.csv): ")

df_normal = pd.read_csv(input_file_name)
df_anomalous = pd.read_csv(input_file_name2)
df_modified = pd.read_csv(input_file_name3)

df_modified.replace('<not', 0, inplace=True)

df_normal.drop(['Time Elapsed (s)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)
df_anomalous.drop(['Time Elapsed (s)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)
df_modified.drop(['Time Elapsed (s)',' RSS (%)',' Virtual Memory Size',' Total Memory Usage (%)'], inplace = True, axis = 1)
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

# Replace NaN values with the mean (you can choose a different strategy)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_test_anomalous = imputer.transform(X_test_anomalous)
X_test_modified = imputer.transform(X_test_modified)
clf = LOF(n_neighbors=50, contamination=contamination_factor)

# Model training
clf.fit(X_train)

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

sns.countplot(x=pred_anomalous, palette='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('Distribution of Predicted Labels (Anomalous Test Set)')
plt.show()

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

sns.countplot(x=pred_modified, palette='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('Count')
plt.title('Distribution of Predicted Labels (Anomalous Test Set)')
plt.show()

# Extracting feature names
feature_names = df_normal.columns

# Function to calculate F1-Score for a given feature or group of features
def calculate_f1_score(feature_columns):
    # Extract the selected features
    X_train_subset = X_train_scaled[feature_columns]
    X_test_subset = X_test_scaled[feature_columns]
    X_test_anomalous_subset = X_test_anomalous_scaled[feature_columns]
    X_test_modified_subset = X_test_modified_scaled[feature_columns]

    # Model training
    clf.fit(X_train_subset)

    # Model evaluation
    pred_test = clf.predict(X_test_subset)
    f1_test = f1_score(y_test, pred_test)

    pred_anomalous = clf.predict(X_test_anomalous_subset)
    f1_anomalous = f1_score(y_anomalous, pred_anomalous)

    pred_modified = clf.predict(X_test_modified_subset)
    f1_modified = f1_score(y_modified, pred_modified)

    return f1_test, f1_anomalous, f1_modified

# Initialize lists to store F1-Scores
f1_scores_test = []
f1_scores_anomalous = []
f1_scores_modified = []

# Iterate through each feature and calculate F1-Scores
for feature in feature_names:
    f1_test, f1_anomalous, f1_modified = calculate_f1_score([feature])
    f1_scores_test.append(f1_test)
    f1_scores_anomalous.append(f1_anomalous)
    f1_scores_modified.append(f1_modified)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(feature_names, f1_scores_test, label='Test Set', alpha=0.5)
ax.bar(feature_names, f1_scores_anomalous, label='Anomalous Test Set', alpha=0.5)
ax.bar(feature_names, f1_scores_modified, label='Modified Test Set', alpha=0.5)

ax.set_ylabel('F1-Score')
ax.set_title('LOF F1-Score for Individual Features')
ax.legend()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

