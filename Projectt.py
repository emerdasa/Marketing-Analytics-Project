# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:24:25 2020

@author: Eden
"""
# %%% 1.
import pandas as pd
import os
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model            import LinearRegression
from sklearn.datasets                import load_iris
from sklearn.model_selection         import train_test_split
from sklearn.naive_bayes             import GaussianNB
from sklearn                         import tree


pd.set_option('display.max_rows',     20)
pd.set_option('display.max_columns',  20)
pd.set_option('display.width',       800)
pd.set_option('display.max_colwidth', 20)

np.random.seed(1)
os.chdir('C:/Users/Almaz/Desktop/Marketing Analytics/Project')


# %%% 2. DATA SPLITTING
dta               = pd.read_excel('C:/Users/Almaz/Desktop/Marketing Analytics/Project/crime data.xlsx').reset_index()
#the above code imports the crime dataset
dta = dta.drop(columns = ['Unnamed: 8', 'Unnamed: 9'])

#Descriptive Statistics 
dta.min()
dta.max()
dta.mean()
dta.std()

dta['ML_group']   = np.random.randint(450,size = dta.shape[0])  
dta               = dta.sort_values(by='ML_group')
inx_train = dta[dta['ML_group']< 270]

# %%% TVT - SPLIT
inx_train = dta[dta['ML_group']< 270]      #splitting the data into train, test and validating
valid = (dta['ML_group']< 320) & (dta['ML_group']>= 270)
inx_valid = dta.loc[valid, :] 
inx_test  = dta[dta['ML_group']>= 320] 

#defining X and Y
X       = [dta['bachelor degree-percentage-25-64 values'],	
           dta['per capita personal income'],
           dta['unemployment rate-percentage'],	
           dta['police presence per 100,000 population'], 
           dta['death penality']]

Y = dta['property crime'] 

# in R lm(Y~X)
Y_train = inx_train['property crime'].to_list()   # Y is dependent variable and X is all of the independent variables
Y_valid = inx_valid['property crime'].to_list()
Y_test  = inx_test['property crime'].to_list()


X_train = inx_train.iloc[:, 4:]
X_valid = inx_valid.iloc[:, 4:]
X_test  = inx_test.iloc[:, 4:]

# %%% Plots

import matplotlib.pyplot as plt

#Plot for Education and Property Crime 
N = 450
trend = np.polyfit(dta['bachelor degree-percentage-25-64 values'],dta['property crime'],1)
plt.plot(dta['bachelor degree-percentage-25-64 values'],dta['property crime'],'o')
trendpoly = np.poly1d(trend) 
plt.plot(dta['bachelor degree-percentage-25-64 values'],trendpoly(dta['bachelor degree-percentage-25-64 values']))
plt.xlabel('Percentage of Population with a Bachelor Degree')
plt.ylabel('Property Crime')


#Plot for Income and Property Crime 
N = 450
trend = np.polyfit(dta['per capita personal income'],dta['property crime'],1)
plt.plot(dta['per capita personal income'],dta['property crime'],'o')
trendpoly = np.poly1d(trend) 
plt.plot(dta['per capita personal income'],trendpoly(dta['per capita personal income']))
plt.xlabel('Per Capita Personal Income')
plt.ylabel('Property Crime')
 
#Plot for Unemployement and Property Crime 
N = 450
trend = np.polyfit(dta['unemployment rate-percentage'],dta['property crime'],1)
plt.plot(dta['unemployment rate-percentage'],dta['property crime'],'o')
trendpoly = np.poly1d(trend) 
plt.plot(dta['unemployment rate-percentage'],trendpoly(dta['unemployment rate-percentage']))
plt.xlabel('Unemployement Rate')
plt.ylabel('Property Crime')


#Plot for Police Presence and Property Crime 
N = 450
trend = np.polyfit(dta['police presence per 100,000 population'],dta['property crime'],1)
plt.plot(dta['police presence per 100,000 population'],dta['property crime'],'o')
trendpoly = np.poly1d(trend) 
plt.plot(dta['police presence per 100,000 population'],trendpoly(dta['police presence per 100,000 population']))
plt.xlabel('Police per 100,000 population')
plt.ylabel('Property Crime')


#Plot for Death Penality and Property Crime 
N = 450
trend = np.polyfit(dta['death penality'],dta['property crime'],1)
plt.plot(dta['death penality'],dta['property crime'],'o')
trendpoly = np.poly1d(trend) 
plt.plot(dta['death penality'],trendpoly(dta['death penality']))
plt.xlabel('Death Penality')
plt.ylabel('Property Crime')


# %%%%  Linear Regression Model
model  = LinearRegression()
clf = model.fit(X_train, Y_train)    #no need to do a forloop b/c linear regression doesn't have hyper perameters
Y_pred = clf.predict(X_test)                   


#%%%% Standard Deviation for Linear Regression Model
from sklearn.metrics import mean_squared_error
import sklearn
import numpy as np

np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred)) 

#The above code shows the bais or standard deviation of the linear regression model
#Confusion matrix couldn't be done because continious is not supported 

# %%% LASSO Model
from sklearn import linear_model
import sklearn

## The below code runs a forloop and graphs the result to show which alpha gives the lowest mean squared error
n_alpha = [0.001, 0.01, 0.1, 1, 10, 100]
tes =[]
for i in n_alpha:
    clf = linear_model.Lasso(alpha=i)      #the penality factor      
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    tes.append(mse)

    print(clf.coef_)
    print(clf.intercept_)
 
    
import matplotlib.pyplot as plt    
plt.plot(tes)
plt.xlabel('Alphas') 
plt.ylabel('Mean Squared Error') 
plt.legend('test')
   
#Since the code above shows that 0.001 alpha gives the smallest mean squared error, the below code uses that alpha to predict Y

clf = linear_model.Lasso(alpha=0.001)      #the penality factor      
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

print(clf.coef_)
print(clf.intercept_)

#%%% Standard Deviation for Lasso Model

np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred)) 

#%%%% Random Forest Regressor Model

from sklearn.ensemble import RandomForestRegressor


n_estimators = [5, 10, 20, 40, 80]
tes =[]
for i in n_estimators:
    clf = RandomForestRegressor(n_estimators=i)            
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
    tes.append(mse)

 
import matplotlib.pyplot as plt    
plt.plot(tes)
plt.xlabel('Estimators') 
plt.ylabel('Mean Squared Error') 
plt.legend('test')
   

clf = RandomForestRegressor(n_estimators=5)            
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)

#%%%% Standard Deviation for Random Forest Regressor Model

np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred)) 

#This model gives the lowest standard deviation compared to all of the other models
#This means that it gives us a prediction that a better accuracy than the other models
#This shows that there may be some non linear relationship within the data
#and since this model is flexible, it does a good job of explaining the relationship between the variables in the data and predicting Y. 

# %%% Naive Bayes Classification Model
clf                              = GaussianNB()
result_nb                        = clf.fit(X_train, Y_train) 
Y_pred = result_nb.predict(X_test) 

         
#%%%% Standard Deviation for Naive Bayes Model

np.sqrt(sklearn.metrics.mean_squared_error(Y_test, Y_pred)) 

