# -*- coding: utf-8 -*-
"""
Created on Tue May 29 21:27:57 2018

@author: jamespatten

Data Extraction, Transformation & Modelling
"""

#------Imports------# 
#pipeline
import pandas as pd
from sklearn import preprocessing
import re
import csv
import time
import geocoder
from geopy.geocoders import Nominatim
import datetime

#webscraper
from bs4 import BeautifulSoup
import requests
from requests import get
#calc tools
import numpy as np
from statistics import median
from requests.exceptions import RequestException
from contextlib import closing

#databse tools 
from sqlalchemy import create_engine
import pymysql


#frontend/client tools
from geojson import Feature, Point, FeatureCollection
import json
import websockets
import asyncio

#------Set-up Data Drame------#
df = pd.DataFrame()

#------Web Scraper/Data Extraction ------#
urls = []
hyperlinks = []
housePages = []

for i in range(1,5): #max is 101
    url = 'https://www.zoopla.co.uk/for-sale/property/london/?identifier=london&page_size=100&q=London&search_source=refine&radius=0&pn='+ str(i)
    urls.append(url)
for url in urls:
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    
    #Scraping House Page data
    for hyperlink in soup.find_all('a', {'class': 'listing-results-address'}):
        hyperlinks.append(hyperlink['href'])
    for link in hyperlinks:
        house_page = 'https://www.zoopla.co.uk' + str(link)
        housePages.append(house_page)

#lists to hold data for dataframe
prices = []
addresses = []
descs = []
infos = []
closestSchool_distance = []
closestStation_distance = []
mbps = []
noBedrooms = []
noBathrooms = []

for page in housePages:
    r = requests.get(page)
    soupPage = BeautifulSoup(r.content, 'lxml')
    
    #scrape price
    price = soupPage.find('p', {"class":"ui-pricing__main-price"})
    if price is not None:
        prices.append(price.text)
    else:
        prices.append('None')
    
    #scrape address
    address = soupPage.find('h2', {"class":"ui-prop-summary__address"})
    if address is not None:
        addresses.append(address.text)
    else:
        addresses.append('None')
    
    #short desc
    desc = soupPage.find('h1', {"class":"ui-prop-summary__title ui-title-subgroup"})
    if desc is None:
        descs.append('None')
    else:
        descs.append(desc.text)
    
    #scrape the property long desc - need to add a case when info isn't available
    info = soupPage.find('div',{"class":"dp-description__text"})
    if info is None:
        infos.append('None')
    else:
        infos.append(info.text)
        
    #scrape the distance to nearest school and train station
    amenities_distance = soupPage.find_all('span',attrs={"class":"ui-local-amenities__distance"})
    amenityDistances = []
    for distance in amenities_distance:
        distance = distance.text
        distance = distance.replace(' miles','')
        amenityDistances.append(distance)
    amenityDistances = amenityDistances[:4]
    schoolDistances = amenityDistances[:2]
    stationDistances = amenityDistances[2:]
    closestSchool_distance.append(min(schoolDistances))
    closestStation_distance.append(min(stationDistances))
#    if len(schoolDistances) == 0:
#        closestSchool_distance.append(np.nan)
#    else:
#        closestSchool_distance.append(min(schoolDistances))
#    if len(amenityDistances) == 0:
#        closestStation_distance.append(np.nan)
#    else:
#        closestStation_distance.append(min(stationDistances))
        
    #scrape broadband speed 
    speed = soupPage.find_all('p',attrs={"class":"dp-broadband-speed__wrapper-text"})
    if speed == []:
        speed = 'None'
        mbps.append(speed)
    else:
        mbps.append(speed[-1].text)
        
    #scrape number of bedrooms/bathroom

    amenityNo = soupPage.find_all('li',attrs={"class":"ui-list-icons__item"})
    for item in amenityNo:
        if 'bedroom' in item.text:
            noBedrooms.append(item.text)
        if 'bathroom' in item.text:
            noBathrooms.append(item.text)     
    if any('bedroom' in item.text for item in amenityNo) == False:
            noBedrooms.append('None')
    if any('bathroom' in item.text for item in amenityNo) == False:
            noBathrooms.append('None')   
      
#------Data Wrangling/Feature Engineering------#

#price
df['price'] = prices
df['price'] = df['price'].str.extract('(\d+([\d,]?\d)*(\.\d+)?)', expand=True) #remove extract numbers with commas
df['price'] = df['price'].str.replace(',','')
medianPrice = [] #replace Null values with median of ingested data
for i in df['price']:
    if pd.isnull(i) == False:
        medianPrice.append(i)
medianPrice = list(map(int, medianPrice))
df['price'] = df['price'].fillna(median(medianPrice))
df['price'] = df['price'].astype('int64') 

#addresses
addresses = [address.replace('\n', '') for address in addresses]
df['address'] = pd.Series(addresses)
df['address'] = df['address'].str.replace(',','')

##postcode
#postcodes = []
#for address in addresses:
#    words = address.split()  # list of words
#    postcode = words[-1]
#    postcodes.append(postcode)
#df['postcode'] = postcodes

#areas
#boroughs = []
#for address in addresses:
#    words = address.split(",")
#    words = words[-1].split(' ')
#    drop = [i for i,y in enumerate(words) if (bool(re.search(r'\d', y))) == True or (y =='') or (y=='...')]
#    for i in sorted(drop, reverse=True):
#        del words[i]
#    borough = ' '.join(words)
#    boroughs.append(borough)
#df['borough'] = pd.Series(boroughs)
#df_boroughs = pd.get_dummies(df, columns=["borough"])
#df = [df, df_boroughs]

#geocodes
#geocodes = [] #convert address to geocode so that it can be visualised on a map
#for i in df['address']:
#    g = geocoder.google(str(i))
#    geocode = g.latlng
#    geocodes.append(geocode)
#    geocodes = [[0,0] if x==None else x for x in geocodes]
##df['geocode'] = geocodes 
#
#lat, lng = zip(*geocodes)


geolocator=Nominatim(timeout=10, user_agent = "houses")
lat = []
lng = [] 
for i in df['address']:
    g = geolocator.geocode(str(i))
    if g == None:
        lat.append(0)
        lng.append(0)
    else:
        lat.append(g.latitude)
        lng.append(g.longitude)
df['lat'] = lat
df['lng'] = lng

#lat long seperates


#number of bedrooms
df['nobed'] = noBedrooms
df['nobed'] = df['nobed'].str.extract('(\d+([\d,]?\d)*(\.\d+)?)', expand=True)
medianNoBed = [] #replace Null values with median of ingested data
for i in df['nobed']:
    if pd.isnull(i) == False:
        medianNoBed.append(i)
medianNoBed = list(map(int, medianNoBed))
df['nobed'] = df['nobed'].fillna(median(medianNoBed))

#number of bathrooms
df['nobath'] = noBathrooms
df['nobath'] = df['nobath'].str.extract('(\d+([\d,]?\d)*(\.\d+)?)', expand=True)
medianNoBath = [] #replace Null values with median of ingested data
for i in df['nobath']:
    if pd.isnull(i) == False:
        medianNoBath.append(i)
medianNoBath = list(map(int, medianNoBath))
df['nobath'] = df['nobath'].fillna(median(medianNoBath))

#closest school
closestSchool_distance = list(map(float, closestSchool_distance))
df['closestSchool'] = closestSchool_distance
#medianClosestSchool = [] #replace Null values with median of ingested data
#for i in df['closestSchool']:
#    if pd.isnull(i) == False:
#        medianClosestSchool.append(i)
#medianPrice = list(map(float, medianClosestSchool))
#df['closestSchool'] = df['closestSchool'].fillna(median(medianClosestSchool))

#closest station
closestStation_distance = list(map(float, closestStation_distance))
df['closestStation'] = closestStation_distance
#medianClosestStation = [] #replace Null values with median of ingested data
#for i in df['closestStation']:
#    if pd.isnull(i) == False:
#        medianClosestStation.append(i)
#medianPrice = list(map(float, medianClosestStation))
#df['closestStation'] = df['closestStation'].fillna(median(medianClosestStation))

#broadband speed - mbps
df['mbps'] = mbps
df['mbps'] = df['mbps'].str.extract('(\d+([\d,]?\d)*(\.\d+)?)', expand=True) 
medianMbps = [] #replace Null values with median of ingested data
for i in df['mbps']:
    if pd.isnull(i) == False:
        medianMbps.append(i)
medianMbps = list(map(float, medianMbps))
df['mbps'] = df['mbps'].fillna(median(medianMbps))

#housetype
start = 'bed'
end = 'for'
housetypes = []
for x in descs:
    #y = x.replace(' ','')
    housetype = x[x.find(start)+len(start):x.rfind(end)]
    housetype = housetype.strip()
    if housetype == 'Studio':
        housetype = 'flat'
    if housetype == 'end terrace house':
        housetype = 'terraced house'
    if housetype == 'Property' or housetype == 'property':
        housetype = 'flat'
    housetypes.append(housetype)
df['type'] = housetypes 

#datetime
dates = []
for x in descs:
    dates.append(str(datetime.datetime.now()))
df['datetime'] = dates
    
#Write into dataset using SQL Alchemy
engine = create_engine("mysql+pymysql://root:root@localhost:3306/houses")
df.to_sql(name= 'houses', con=engine, if_exists='append', index=False)

