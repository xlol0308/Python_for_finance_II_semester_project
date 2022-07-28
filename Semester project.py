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
import sklearn.metrics as metr
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
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
#data = data[data.index.year<2022]
df = data.resample('d').last()


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
return_data = (1+df["Return"].ffill().dropna())*100
n=15
d = 16
l_v  = sp.svr_retrain(price_data,n=n,p=1,d=d,kernel='linear')
r_v = sp.svr_retrain(price_data,n=n,p=1,d=d,kernel='rbf')
p_v = sp.svr_retrain(price_data,n=n,p=1,d=d,kernel='poly')
s_v = sp.svr_retrain(price_data,n=n,p=1,d=d,kernel='sigmoid')
#%%

d = 60
l_r  = sp.svr_retrain(return_data,n=n,p=1,d=d,kernel='linear')
r_r = sp.svr_retrain(return_data,n=n,p=1,d=d,kernel='rbf')
p_r = sp.svr_retrain(return_data,n=n,p=1,d=d,kernel='poly')
s_r = sp.svr_retrain(return_data,n=n,p=1,d=d,kernel='sigmoid')
#%%
base_p = [l_v,r_v,p_v,s_v]
base_r = [l_r,r_r,p_r,s_r]

a = pd.DataFrame(columns = ["Pr linear","Pr rbf","Pr poly","Pr sigmoid"])
b = pd.DataFrame(columns = ["Ret linear","Ret rbf","Ret poly","Ret sigmoid"])

for j in range(4):
    a[a.columns[j]] = sp.metrics(base_p[j],df.iloc[-n:,3])[0]
    b[b.columns[j]] = sp.metrics(base_r[j],(1+df.iloc[-n:,-1])*100)[0]

stat_metr = pd.concat([a,b],axis=1)

#print(stat_metr)


#%%

df2 = data[["Close"]]#[data.index.year >= 2022]
#df2 = df2[df2.index.month>3]
df2["Date1"] = df2.index.date
df2["Time"] = df2.index.time


price_data_hf = pd.pivot_table(
    data=df2,
    index = "Date1",
    columns = "Time",
    values = "Close"
).ffill().dropna()

return_data_hf = np.log(price_data_hf/price_data_hf.shift())

return_data_hf =(1+return_data_hf)*100

#%%
n=14
p=7
d = int((n/3)*4)
pred_hf_price = sp.svr_retrain_mul(price_data_hf,n=n,d=d,p=p)


pred_hf_return = sp.svr_retrain_mul(return_data_hf,n=n,d=d,p=p,kernel="sigmoid")

#print(mae_hf_p,mae_hf_r)

#%%


pred_pr = sp.svr_retrain(price_data,n=n,p=p,d=d,kernel='rbf')
pred_r = sp.svr_retrain(return_data,n=n,p=p,d=d,kernel='poly')


df_metr = pd.DataFrame(columns = ["Price","HF Price","Return","HF Return"])
df_metr["Price"] = sp.metrics(pred_pr,df.iloc[-n:,3])[0]
df_metr["HF Price"] = sp.metrics(pred_hf_price,df.iloc[-n:,3])[0]
df_metr["Return"] = sp.metrics(pred_r,(1+df.iloc[-n:,-1])*100)[0]
df_metr["HF Return"] = sp.metrics(pred_hf_return,return_data_hf.iloc[-n:,-1])[0]
print(df_metr)



#%%



n= 7
p=1
base_p = sp.rfr(price_data,n=n,p=p)
base_r = sp.rfr(return_data,n=n,p=p)


met = sp.metrics(base_p,price_data.iloc[-n:])
df_metr_rf = pd.DataFrame(columns = ["Price","Return"])
df_metr_rf["Price"] = sp.metrics(base_p,df.iloc[-n:,3])[0]

df_metr_rf["Return"] = sp.metrics(base_r,(1+df.iloc[-n:,-1])*100)[0]
                            
print(df_metr_rf)
