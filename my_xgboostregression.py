import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

# Loading and Exploring the Data

warnings.filterwarnings("ignore")


diamonds = sns.load_dataset("diamonds")

diamonds.head()

# print(diamonds.describe())

# build the learning frame
from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = diamonds.drop('price', axis=1), diamonds[['price']]
# Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')
import xgboost as xgb
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=1)
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
# actual=
import numpy as np

# mse = np.mean((actual - predicted) ** 2)
# rmse = np.sqrt(mse)

# params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
# n = 10000
#
# evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
#
# model = xgb.train(
#    params=params,
#    dtrain=dtrain_reg,
#    num_boost_round=n,
#    evals=evals,
#    verbose_eval=10,
#    early_stopping_rounds=50
# )

params = {"objective": "reg:squarederror", "tree_method": "gpu_hist","subsample":0.5}
n = 1000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n,
   nfold=5,

   early_stopping_rounds=2
)
print(results.head())
best_rmse1 = results['test-rmse-mean']
best_rmse2=results["train-rmse-mean"]
print(best_rmse1,best_rmse2)
plt.plot(best_rmse2,best_rmse1)
plt.show()
1
