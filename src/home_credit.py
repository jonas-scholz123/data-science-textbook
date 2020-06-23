'''
From https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
'''



# numpy and pandas for data manipulation
import numpy as np
import pandas as pd

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Load Data

train = pd.read_csv("../data/home_credit/application_train.csv")
test  = pd.read_csv("../data/home_credit/application_test.csv")

train_labels = train["TARGET"]

#%%

# Handle categoricals
train.dtypes.value_counts()
train.select_dtypes("object").apply(pd.Series.nunique, axis = 0)

#%%
# Label encode categoricals with 2 or less unique values

le = LabelEncoder()
le_count = 0

for col in train:
    if train[col].dtype == "object":
        if len(set(train[col])) <= 2:
            le.fit(train[col])

            train[col] = le.transform(train[col])
            test[col] = le.transform(test[col])

            le_count += 1

# one hot encoding:

train = pd.get_dummies(train)
test = pd.get_dummies(test)

print("%d columns were label encoded" % le_count)

#%% align one-hot columns:

train, test = train.align(test, join = "inner", axis = 1)

train["TARGET"] = train_labels

assert train.shape[1] == test.shape[1] + 1

#%%
# filter anomalies

#train["DAYS_EMPLOYED"].describe()
## Create an anomalous flag column
#train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243
## Replace the anomalous values with nan
#train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

#%% Correlations
#
#correlations = train.corr()["TARGET"].sort_values()
#
##%%
#correlations.head(20)
#correlations.tail(20)

#%% Logistic Regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

X = train.copy()
X = X.drop(columns = "TARGET")

y = test.copy()
imputer = SimpleImputer(strategy = "median")

imputer.fit(X)

X = imputer.transform(X)
y = imputer.transform(y)

#Normalise

scaler = MinMaxScaler(feature_range = (0, 1))

scaler.fit(X)

X = scaler.transform(X)
y = scaler.transform(y)

log_reg = LogisticRegression()

#train

log_reg.fit(X, train_labels)

#%%

log_reg_predictions = log_reg.predict_proba(y)
default_probs = log_reg_predictions[:, 1]

yhat = test[["SK_ID_CURR"]]

yhat["PredDefault"] = default_probs

yhat.to_csv("../submissions/log_reg_baseline.csv")
