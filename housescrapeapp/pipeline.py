# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:27:57 2018
@author: jamespatten
"""

#------Imports------# 
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re
import csv
import time
import geocoder
from datetime import datetime
from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import requests

#databse tools 
from sqlalchemy import create_engine

#Modelling 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

#------Set-up Data Drame------#)
df = pd.DataFrame()

#------Web Scraper/Data Extraction ------#
urls = []
prices = []
addresses = []
infos = []
description = []
start = time.time()
for i in range(1,2): #max is 101
    url = 'https://www.zoopla.co.uk/for-sale/property/london/?identifier=london&page_size=100&q=London&search_source=refine&radius=0&pn='+ str(i)
    urls.append(url)
for url in urls:
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    for price in soup.find_all('a', {"class":"listing-results-price text-price"}):
        prices.append(price.text) 
    for address in soup.find_all('a', {'class': 'listing-results-address'}):
        addresses.append(address.text)
    for info in soup.find_all('p',attrs={"class":None}):
        infos.append(info.text)  
    for descs in soup.find_all('h2', {'class':'listing-results-attr'}):
        desc = descs.find('a').contents[0]
        description.append(desc)

#------Data Wrangling/Feature Engineering------#
#price
df['price'] = prices
df['price'] = df['price'].str.extract('(\d+([\d,]?\d)*(\.\d+)?)', expand=True) #remove extract numbers with commas
df['price'] = df['price'].str.replace(',','')
df['price'].dropna(how='any', inplace=True)
df['price'] = df['price'].astype('int64') 
df['price'].dropna(how='any', inplace=True)    
df['price'].isnull().values.any()
#adresses
addresses = [address.replace('\n', '') for address in addresses]
df['address'] = pd.Series(addresses)
df['address'] = df['address'].str.replace(',','')
#postcode
postcodes = []
for address in addresses:
    words = address.split()  # list of words
    postcode = words[-1]
    postcodes.append(postcode)
df['postcode'] = postcodes
#boroughs
boroughs = []
for address in addresses:
    words = address.split(",")
    words = words[-1].split(' ')
    drop = [i for i,y in enumerate(words) if (bool(re.search(r'\d', y))) == True or (y =='') or (y=='...')]
    for i in sorted(drop, reverse=True):
        del words[i]
    borough = ' '.join(words)
    boroughs.append(borough)
    le = preprocessing.LabelEncoder()
    le.fit(boroughs)
    le.classes_
    boroughcode = le.transform(boroughs)
df['borough'] = pd.Series(boroughs)
df['boroughcode']  = boroughcode
df['borough'].value_counts()
#geocodes
geocodes = [] #convert address to geocode so that it can be visualised on a map
for i in df['address']:
    g = geocoder.google(str(i))
    geocode = g.latlng
    geocodes.append(geocode)
    geocodes = [[0,0] if x==None else x for x in geocodes]
df['geocode'] = geocodes 
#lat long seperates
lat, lng = zip(*geocodes)
df['lat'] = lat
df['lng'] = lng


#number of bedrooms
nobeds = []
for x in description:
     bed = x.split()[0]
     if bed == 'Studio':
         bed = 1
     nobeds.append(bed)
df['nobed'] = pd.Series(nobeds)
#housetype
start = 'bed'
end = 'for'
housetypes = []
for x in description:
    y = x.replace(' ','')
    housetype = y[y.find(start)+len(start):y.rfind(end)]
    housetypes.append(housetype)
df['type'] = pd.Series(housetypes)
df['type'].dropna(how='any', inplace=True) 
df['type'].value_counts() #count of distinct values - need to redo this cause getting NaN values
df['type'] = df['type'].map( {'flat': 0, 'terracedhouse': 1, 'semi-detachedhouse': 2, 'property': 3, 'maisonette': 4, 'endterracehouse': 1, 'detachedhouse': 5, 'udio': 6, 'bungalow': 7, 'mewshouse':8, 'link-detachedhouse':5, 'semi-detachedbungalow':9,'townhouse':10, 'rracedhouse':1,'tachedhouse':5})
#df = df.drop(['desc'], axis=1)

#---further data preprocessing---#
#create an id column

df = df.dropna()

#df["id"] = df.index + 1

#-----EDA-----#
#mean = np.asarray(df.iloc[:,0], dtype=np.float).mean()
#pd.DataFrame(np.asarray(valores, dtype=np.float))

#----Prediction Model----#
model_cols = ['price','boroughcode','nobed','type']
df_model = df[model_cols]
X = df_model.drop("price", axis=1)
y = df_model["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

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
#Neural Networks
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp.predict(X_test)
mlp_score = mlp.score(X_test, y_test)
print(mlp_score)

#output prediction based on parameters
#How to do it in R
input = pd.DataFrame(columns=('boroughcode','nobed','type'))
input.at[1, 'boroughcode'] = 6
input.at[1, 'nobed'] = 3
input.at[1, 'type'] = 4
print(tree.predict(input))

end=time.time()
#time_elapsed = end - start
#print(time_elapsed)

#------Outputs ------#
engine = create_engine("mysql+pymysql://root:root@localhost:3306/houses")
df_sql = df.drop('geocode',axis =1)
df_sql.to_sql(name= 'houses', con=engine, if_exists='append', index=False)

#need to think of a duplicate management system


#with open('index.csv', 'a') as csv_file:
#writer = csv.writer(csv_file)
#writer.writerow([search.listing, datetime.now()])
#writer.writerow([search.listing, datetime.now()])
