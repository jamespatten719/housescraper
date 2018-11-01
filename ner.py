# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 20:58:07 2018

@author: jamespatten

NER Tagger for User Input String
Built using tutorial by Tobias Sterbak
https://www.depends-on-the-definition.com/sequence-tagging-lstm-crf/


"""

#import asyncio
#import websockets
from flask import Flask


import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional

#import matplotlib.pyplot as plt
#plt.style.use("ggplot")


#Implementing an LSTM
data = pd.read_csv("ner_dataset.csv", encoding="latin1")

words = list(set(data["Word"].values))
words.append("ENDPAD")
n_words = len(words); n_words

tags = list(set(data["Tag"].values))
n_tags = len(tags); n_tags

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

sentences = getter.sentences

#plt.hist([len(s) for s in sentences], bins=50)
#plt.show()


#For the use of neural nets we need to use equal-lenght input sequences. 
#So we are going to pad our sentences to a length of 50. 
#But first we need dictionaries of words and tags.

max_len = 50
word2idx = {w: i for i, w in enumerate(words)} #giving each word an id
tag2idx = {t: i for i, t in enumerate(tags)} #giving each tag an id

#map the senctences to a sequence of numbers and then pad the sequence.
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

#For training the network we also need to change the labels y to categorial.
y = [to_categorical(i, num_classes=n_tags) for i in y]

#split dataset
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

#building the layers of the neural network
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer
model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=5, validation_split=0.1, verbose=1)

filename = 'ner.sav'
model.save(filename)


#hist = pd.DataFrame(history.history)
#
#plt.figure(figsize=(12,12))
#plt.plot(hist["acc"])
#plt.plot(hist["val_acc"])
#plt.show()
#
##testing some predictions
##use this model to post a new sentence
i = 2318
p = model.predict(np.array([X_te[i]]))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(X_te[i], p[0]):
#    print("{:15}: {}".format(words[w], tags[pred]))
#    
##predict from an input sentence
##check if sentence in word2idx
##need to add 
#
client_sentence = ''
client_sentence = client_sentence.split()
#
#test_sentence = []
#for i in client_sentence:
#    if i in word2idx.keys():
#        test_sentence.append(word2idx[i]) #this hasn't appended all the keys
#    else:
#        print('not in dict')
#
##pad test sentence_
test_sentence_nested = [[i] for i in client_sentence ]
test_sentence_padded = pad_sequences(maxlen=max_len, sequences=test_sentence_nested, padding="post", value=n_words - 1)
        
p = model.predict(np.array(test_sentence_padded))
p = np.argmax(p, axis=-1)
print("{:15} ({:5}): {}".format("Word", "True", "Pred"))
for w, pred in zip(test_sentence , p[0]):
    print("{:15}: {}".format(words[w], tags[pred]))
    

