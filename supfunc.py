#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:43:48 2022

@author: fox2

Supprting functions 
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
#%%


# plotting fuctions 
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)
    
    # Calculate the array of the variances of the lagged differences
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    
    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
def plot_correlogram(x, lags=None, title=None):    
    lags = min(10, int(len(x)/5)) if lags is None else lags
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    x.plot(ax=axes[0][0])
    q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
    stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f} \nHurst: {round(hurst(x.values),2)}'
    axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
    stat.probplot(x, plot=axes[0][1])
    mean, var, skew, kurtosis = stat.moment(x, moment=[1, 2, 3, 4])
    s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
    axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
    plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
    plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
    axes[1][0].set_xlabel('Lag')
    axes[1][1].set_xlabel('Lag')
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    
    
#%%
#Model construction 
#moving windowprediction
#SVR
def svr_retrain(df,n=15,p=2,d=60,kernel='rbf'):
    
    pred_v = np.empty(0)
    d=d+n+p
    df = df.iloc[-d:]
    n_row = df.shape[0]

    for i in range(0,n,p):
        
        samp = df.iloc[i:n_row-n+i]
        #CREATE THE INDEPENDENT DATA SET (X)
        samp_pred = samp.shift(-p)
        #print(data)
        # Convert the dataframe to a numpy array and drop the prediction column
        X = np.array(samp).reshape(-1, 1)
    
        #Remove the last 'n' rows where 'n' is the prediction_days
        #print(samp)
        X = X[:len(samp)-p]
        
        #CREATE THE DEPENDENT DATA SET (y) 
        # Convert the dataframe to a numpy array (All of the values including the NaN's) 
        y = np.array(samp_pred).reshape(-1, 1) 
        # Get all of the y values except the last 'n' rows 
        y = y[:-p] 
        
        
        # Create and train the Support Vector Machine 
        svr_rbf = SVR(kernel=kernel, C=1e3, gamma=0.00001)#Create the model
        svr_rbf.fit(X, y) #Train the model
        
        # Testing Model: Score returns the accuracy of the prediction. 
        # The best possible score is 1.0
        svr_rbf_confidence = svr_rbf.score(X, y)
        #print("svr_rbf accuracy: ", svr_rbf_confidence)
        
        # Print the predicted value
        svm_prediction = svr_rbf.predict(X[-p:])
        pred_v = np.append(pred_v,svm_prediction)
        
        
    return  pred_v
#%%
def RMSE(x,y):
    x = np.array(x)
    y = np.array(y)

    result = mean_squared_error(x,y)
    return np.sqrt(result)
    