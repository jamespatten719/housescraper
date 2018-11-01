# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 19:40:40 2018

@author: jamespatten
"""

#Database tools
from sqlalchemy import create_engine
import pymysql
import pandas as pd
import pickle

#Modelling 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

connection = pymysql.connect(host='localhost', port=3306, user='root', db='houses',password='root')   
df = pd.read_sql('SELECT * FROM HOUSES', connection)

df= pd.get_dummies(df, columns=["type"])
df_model = df.drop(['house_id','address','lat','lng','datetime'],axis=1)
#model_cols = ['price','nobed','nobath','mbps','closestSchool','closestStation']
#df_model = df[model_cols]
X = df_model.drop("price", axis=1)
y = df_model["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)


#Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)
reg.predict(X_test)
reg_score = reg.score(X_test, y_test)
print(reg_score)

#Decision Trees
tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
tree.predict(X_test)
tree_score = tree.score(X_test, y_test)
print(tree_score)

#Random Forrest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf.predict(X_test)
rf_score = rf.score(X_test, y_test)
print(rf_score)

#GRID SEARCH for paramter tuning 
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10],
    'criterion' :['mse', 'mae']
}

CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
CV_rf.fit(X_train, y_train)
CV_rf.best_params_

#Neural Networks
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp.predict(X_test)
mlp_score = mlp.score(X_test, y_test)
print(mlp_score)

filename = 'pricing_model.sav'
pickle.dump(rf, open(filename, 'wb'))
 