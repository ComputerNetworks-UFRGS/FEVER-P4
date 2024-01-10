from sklearn.model_selection import train_test_split
from scipy import stats
from pyod.models.iforest import IForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from pyod.models.ocsvm import OCSVM

input_file_name = input("Enter the input CSV file name (e.g., name.csv): ")
input_file_name2 = input("Enter the anomalous input CSV file name (e.g., name.csv): ")
input_file_name3 = input("Enter the modified input CSV file name (e.g., name.csv): ")

df_normal = pd.read_csv(input_file_name)
df_anomalous = pd.read_csv(input_file_name2)
df_modified = pd.read_csv(input_file_name3)

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

clf = IForest(random_state=42, contamination=contamination_factor)

# Model training
clf.fit(X_train)

# Model evaluation with good behaviour
pred=clf.predict(X_test)
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

# Model evaluation with good behaviour
pred=clf.predict(X_train)
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

# Model evaluation with anomalous behaviour
pred=clf.predict(X_test_anomalous)
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)

# Model evaluation with modified behaviour
pred=clf.predict(X_test_modified)
unique_elements, counts_elements = np.unique(pred, return_counts=True)
print("\t",unique_elements,"    ",counts_elements)
