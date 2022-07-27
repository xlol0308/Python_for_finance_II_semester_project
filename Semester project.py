#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:37:24 2022

@author: fox2
"""
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import seaborn as sns
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import supfunc as sp

#%%
#Plots aesthetics

#%matplotlib inline
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)
rcParams['figure.figsize'] = 16,8


    
#%%

data = pd.DataFrame()
data = pd.read_csv("/Users/fox2/Documents/Python for Finance/Python for Finance II/Semester Project /BTC_high_freq.csv",index_col="Date")
#data["SPY"] = pdr.get_data_yahoo("SPY", start="17/09/2014")["Adj Close"]
data.index = pd.to_datetime(data.index)
data.Close = data.Close.ffill()

df = data.resample('d').last()
data["Return"] = np.log(data["Close"]/data["Close"].shift())


print(data.isna().sum())
print(df.isna().sum())
#%%

#Sampling
df["Return"] = np.log(df["Close"]/df["Close"].shift())
(df["Return"]).cumsum().plot()


price = df.Close.ffill()

sp.plot_correlogram(price)

returns = df.Return.ffill().dropna()

sp.plot_correlogram(returns)
print(df[["Close","Return"]].describe())

#%%
price_data = df["Close"].ffill().dropna()
return_data = df["Return"].ffill().dropna()
v  = sp.svr_retrain(price_data,n=1,p=1,d=60,kernel='rbf')
print(v)


r  = sp.svr_retrain(return_data,n=1,p=1,d=60,kernel='rbf')
print(r)
#%%
def RMSE(x,y):
    x = np.array(x)
    y = np.array(y)
    
    error = ((x-y)**2)
    result = np.sqrt(np.mean(error))
    
    return result
print(RMSE(v,df.iloc[-1:,3]))
#%%
i = mean_squared_error(v,df.iloc[-1:,3])
print(np.sqrt(i))
