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
import sklearn.metrics as metr
from sklearn.svm import SVR
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
#%%

#%matplotlib inline
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)
rcParams['figure.figsize'] = 16,8
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
    x.plot(ax=axes[0][0],title="Price")
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

    for i in tqdm(range(0,n,p)):
        
        samp = df.iloc[i:n_row-n+i]
        #CREATE THE INDEPENDENT DATA SET (X)
        
        # Convert the dataframe to a numpy array and drop the prediction column
        X = samp.values.reshape(-1,1)
        scale = StandardScaler()
        #X = scale.fit_transform(X)
        x_train = X[:-(p+p)]
        x_test = X[-(p+p):-p]
       
    
        
        #CREATE THE DEPENDENT DATA SET (y) 
        # Convert the dataframe to a numpy array 
        y = samp.values.reshape(-1,1)
        y_train = y[p:-(p)]
        y_test = y[-(p):]
        
        
        # Create and train the Support Vector Machine 
        svr_rbf = SVR(kernel=kernel, C=1e3, gamma=0.00001)#Create the model
        svr_rbf.fit(x_train, y_train) #Train the model
        
        # Testing Model: Score returns the accuracy of the prediction. 
        # The best possible score is 1.0
        svr_rbf_confidence = svr_rbf.score(x_train, y_train)
        #print("svr_rbf accuracy: ", svr_rbf_confidence)
        
        # Print the predicted value
        svm_prediction = svr_rbf.predict(x_test)
        pred_v = np.append(pred_v,svm_prediction)
        
        
    return  pred_v

#%%

def svr_retrain_mul(df,n=15,p=1,d=30,kernel='rbf'):
    
    pred_v = np.empty(0)
    d=d+n+p
    df = df.iloc[-d:]
    n_row = df.shape[0]

    for i in tqdm(range(0,n,p)):
        
        samp = df.iloc[i:n_row-n+i]
        #CREATE THE INDEPENDENT DATA SET (X)
        samp_pred = samp.iloc[0:,-1]
        
        # Convert the dataframe to a numpy array and drop the prediction column
        X = np.array(samp)
        scale = StandardScaler()
        #X = scale.fit_transform(X)
        x_train = X[:-(p+p)]
        x_test = X[-(p+p):-p]
        
        #CREATE THE DEPENDENT DATA SET (y) 
        # Convert the dataframe to a numpy array (All of the values including the NaN's) 
        y = np.array(samp_pred)
        y_train = y[p:-(p)]
        y_test = y[-(p):]
        
        
        # Create and train the Support Vector Machine 
        svr_rbf = SVR(kernel=kernel, C=1e3, gamma=0.00001)#Create the model
        svr_rbf.fit(x_train, y_train) #Train the model
        
        # Testing Model: Score returns the accuracy of the prediction. 
        # The best possible score is 1.0
        svr_rbf_confidence = svr_rbf.score(x_train, y_train)
        #print("svr_rbf accuracy: ", svr_rbf_confidence)
        
        # Print the predicted value
        svm_prediction = svr_rbf.predict(x_test)
        pred_v = np.append(pred_v,svm_prediction)
        
    return  pred_v
#%%

def rfr(df,n=15,p=1,d=30):
    
    pred_v = np.empty(0)
    d=d+n+p
    df = df.iloc[-d:]
    x_train = df.iloc[:-(n+p+p)].values.reshape(-1,1)
    y_train = df.iloc[p:-(n+p)].values#.reshape(-1,1)
    n_row = df.shape[0]
   
    grid_rf = {
        'n_estimators': [20, 50, 100, 500, 1000],  
        'max_depth': np.arange(1, 31, 1,  dtype=int),  
        'min_samples_split':np.arange(2, 11, 1,  dtype=int), 
        'min_samples_leaf': np.arange(1, 21, 2, dtype=int),  
        'bootstrap': [True, False], 
        'random_state': [1, 2, 7, 35, 42, 49]
        }
        
    model = RandomForestRegressor()
    rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=100)
    rscv_fit = rscv.fit(x_train, y_train)
    bp = rscv_fit.best_params_

    for i in tqdm(range(0,n,p)):
        
        samp = df.iloc[i:n_row-n+i]
        #CREATE THE INDEPENDENT DATA SET (X)
        
        # Convert the dataframe to a numpy array and drop the prediction column
        X = samp.values.reshape(-1,1)
        scale = StandardScaler()
        X = scale.fit_transform(X)
        x_train = X[:-(p+p)]
        x_test = X[-(p+p):-p]
       
    
        
        #CREATE THE DEPENDENT DATA SET (y) 
        # Convert the dataframe to a numpy array 
        y = samp.values.reshape(-1,1)
        y_train = y[p:-(p)]
        y_test = y[-(p):]
        
        random_state = bp['random_state']
        n_estimators = bp['n_estimators']
        min_samples_split = bp['n_estimators']
        min_samples_leaf = bp['n_estimators']
        max_depth = bp['max_depth']
        bootstrap = bp['bootstrap']
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                      max_depth=max_depth, bootstrap=bootstrap)
        
        model_fit = model.fit(x_train,y_train)

        rfr_prediction = model_fit.predict(x_test)
        pred_v = np.append(pred_v,rfr_prediction)
        
    return  pred_v
#%%
def rfr_hf(df,n=15,p=1,d=30):
    
    pred_v = np.empty(0)
    d=d+n+p
    df = df.iloc[-d:]
    x_train = df.iloc[:-(n+p+p)].values
    y_train = df.iloc[p:-(n+p)].iloc[0:,-1].values#.reshape(-1,1)
    n_row = df.shape[0]
   
    grid_rf = {
        'n_estimators': [20, 50, 100, 500, 1000],  
        'max_depth': np.arange(1, 31, 1,  dtype=int),  
        'min_samples_split':np.arange(2, 11, 1,  dtype=int), 
        'min_samples_leaf': np.arange(1, 21, 2, dtype=int),  
        'bootstrap': [True, False], 
        'random_state': [1, 2, 7, 35, 42, 49]
        }
        
    model = RandomForestRegressor()
    rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=100)
    rscv_fit = rscv.fit(x_train, y_train)
    bp = rscv_fit.best_params_

    for i in tqdm(range(0,n,p)):
        
        samp = df.iloc[i:n_row-n+i]
        #CREATE THE INDEPENDENT DATA SET (X)
        
        # Convert the dataframe to a numpy array and drop the prediction column
        X = samp.values.reshape(-1,1)
        scale = StandardScaler()
        X = scale.fit_transform(X)
        x_train = X[:-(p+p)]
        x_test = X[-(p+p):-p]
       
    
        
        #CREATE THE DEPENDENT DATA SET (y) 
        # Convert the dataframe to a numpy array 
        y = samp.values.reshape(-1,1)
        y_train = y[p:-(p)]
        y_test = y[-(p):]
        
        random_state = bp['random_state']
        n_estimators = bp['n_estimators']
        min_samples_split = bp['n_estimators']
        min_samples_leaf = bp['n_estimators']
        max_depth = bp['max_depth']
        bootstrap = bp['bootstrap']
        
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                      max_depth=max_depth, bootstrap=bootstrap)
        
        model_fit = model.fit(x_train,y_train)

        rfr_prediction = model_fit.predict(x_test)
        pred_v = np.append(pred_v,rfr_prediction)
        
    return  pred_v

#%%

def metrics(pred,real):
    mape = round(metr.mean_absolute_percentage_error(pred, real), 3)
    mae = round(metr.mean_absolute_error(pred, real))
    mse = round(metr.mean_squared_error(pred, real))
    rmse =  round(np.sqrt(metr.mean_squared_error(pred, real)))
    r_sq =  round(metr.r2_score(pred, real), 1)
    #print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
    accuracy = round(1 - np.mean(mape),3)
    
    
    df = pd.DataFrame([mape, mae, mse, rmse, r_sq, accuracy],
                      index=["Mean Absolute Prct Error","Mean Absolute Error",
                             "Mean Squared Error","Root Mean Squared Error",
                             "(R squared) Score",'Accuracy'])
    return df


