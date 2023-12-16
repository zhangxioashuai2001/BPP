import seaborn as sns
import pickle
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

# Loading and Exploring the Data

warnings.filterwarnings("ignore")


diamonds = sns.load_dataset("diamonds")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

X, y = diamonds.drop("cut", axis=1), diamonds[['cut']]

# Encode y to numeric
y_encoded = OrdinalEncoder().fit_transform(y)

# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to pd.Categorical
for col in cats:
   X[col] = X[col].astype('category')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)

import xgboost as xgb
# Create classification matrices
dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "multi:softprob", "tree_method": "gpu_hist", "num_class": 5}
n = 1000

results = xgb.cv(
   params, dtrain_clf,
   num_boost_round=n,
   nfold=5,
   metrics=["mlogloss", "auc", "merror"],
)
results.keys()

#
# print(Index(['train-mlogloss-mean', 'train-mlogloss-std', 'train-auc-mean',
#
#       'train-auc-std', 'train-merror-mean', 'train-merror-std',
#
#       'test-mlogloss-mean', 'test-mlogloss-std', 'test-auc-mean',
#
#       'test-auc-std', 'test-merror-mean', 'test-merror-std'],
#
#      dtype='object'))
print(results['test-auc-mean'].max())