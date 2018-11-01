#Flask
import os
from flask import Flask,render_template,request,json

#compiler tools
import pickle
#from keras.models import load_model

#pandas
import pandas as pd

#Database tools
from sqlalchemy import create_engine
import pymysql


#GeoJSONs
from geojson import Feature, Point, FeatureCollection

#Begin App
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('map.html')

@app.route('/geojson', methods=['GET','POST'])
def sendGeo():
    connection = pymysql.connect(host='localhost', port=3306, user='root', db='houses',password='root')
    cur = connection.cursor()
    cur.execute("SELECT LNG, LAT FROM HOUSES")
    sql_geom=list(cur.fetchall())
    
    #create GeoJSON FeatureCollection
    features = []
    for i in sql_geom:
        feature = Feature(geometry=Point(i))
        features.append(feature)
    feature_collection = FeatureCollection(features)
    return json.dumps(feature_collection)

@app.route('/sendSentence', methods=['POST'])
def getSentence():
    sentence =  request.form['sentence'];
    return json.dumps({'status':'OK','sentence':sentence});

@app.route('/sendPrice', methods=['GET','POST'])
def sendPrice():
    filename = 'pricing_model.sav'
    rf = pickle.load(open(filename, 'rb'))
    input = pd.DataFrame(columns=('nobed','nobath','mbps','closestSchool','closestStation'))
    input.at[1, 'nobed'] = 3
    input.at[1, 'nobath'] = 3
    input.at[1, 'mbps'] = 30
    input.at[1, 'closestSchool'] = 0.2
    input.at[1, 'closestStation'] = 0.2
    result = rf.predict(input)
    return pd.Series(result).to_json(orient='values')

#@app.route('/sendTags', methods=['GET','POST'])
#def sendTags():
#    #filename = 'ner.sav'
#    #ner = load_model(filename)
#    model = 'loaded'
#    return model
#    sentence =  request.form['sentence']
#    p = model.predict(np.array(test_sentence_padded))
#    p = np.argmax(p, axis=-1)
#    print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
#    for w, pred in zip(sentence , p[0]):
#        print("{:15}: {}".format(words[w], tags[pred]))
#    return pd.Series(tags).to_json(orient='values')

if __name__=="__main__":
    app.run(debug=True)
