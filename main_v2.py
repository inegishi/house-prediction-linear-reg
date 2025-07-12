import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from IPython.display import display

#from brokenaxes import brokenaxes
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10,6]

import warnings 
warnings.filterwarnings('ignore')


def show_price_histogram():
    plt.figure(figsize=[8,4])
    sns.distplot(
        df["price"],
        bins = 30
    )
    plt.show()


df = pd.read_csv("Housing.csv")

# print(df.head)
df_feat = df.copy()
#Y sets
df_y = df["price"]/1000000

df_feat.drop(columns = ["price"], inplace = True)
feature = [i for i in df_feat.columns]

nu = df[feature].nunique().sort_values()
nf = []
cf = []

for i in range(df_feat.shape[1]):
    if nu.values[i] <= 16:
        cf.append(nu.index[i])
    else:
        nf.append(nu.index[i])

df.drop_duplicates(inplace= True)

#df3 is all the features that are categorical
df3 = df[cf]
for i in cf:
    if df3[i].nunique() == 2:
        df3[i] = pd.get_dummies(df3[i],drop_first=True,prefix=i)
    elif df3[i].nunique() > 2 and df3[i].nunique() <= 17:
        df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df3[i],drop_first=True))],axis=1)


#remove outliers
#df4 is features that are numerical
df4 = df[nf].copy()
print(df4.info)
Q1 = df4.quantile(0.25)
Q3 = df4.quantile(0.75)
IQR = Q3-Q1
# print(Q1)
# print(Q3)
# print(IQR)
for i in nf:
    # print(df.shape[0])
    df4 = df4[
        (df4[i] >= Q1[i] - 1.5 * IQR[i]) &
        (df4[i] <= Q3[i] + 1.5 * IQR[i])
    ]
    # print(df.shape[0])
    
df4 = df4.reset_index(drop = True)
# print(df4.head)

# print(df3.head)
df3 = df3.loc[df4.index].reset_index(drop=True)
df_y = df_y.loc[df4.index].reset_index(drop=True)
# print(df3.shape,df4.shape)
df_feature = pd.concat([df3,df4],axis=1)
# print(df_y.head)
df_feature.columns = df_feature.columns.astype(str)
# print(df_feature.head)
X_train,X_test,y_train,y_test = train_test_split(df_feature, df_y, test_size=0.2, shuffle=True,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
Y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,Y_pred)
r2 = r2_score(y_test,Y_pred)
print(f"mse: {mse}, r2: {r2}")

comparison = pd.DataFrame({
    "actual": y_test,
    "predicted": Y_pred
})
print(comparison.head(50))