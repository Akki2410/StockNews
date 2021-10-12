# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:56:27 2021

@author: aksha
"""
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re

path = "C:/Users/aksha/Desktop/Phython_spyder/NLP/project1_stocknews analysis/Combined_News_DJIA.csv"
df = pd.read_csv(path)


df.head(3)
df.columns



from sklearn.model_selection import train_test_split



train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']
#Xtrain,Xtest,ytrain,ytest = train_test_split(df.drop('Label',1),df['Label'],test_size= 0.2)



#Xtrain.shape
#1591,26
#Xtest.shape
#398,26

# Removing punctuations
data=train.iloc[:,2:27]
test.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
train.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

#data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)

for columns in data:
    data[columns] = data[columns].str.lower()
    
data.head(1)


data.columns
#' '.join(str(x) for x in data.iloc[1,0:25])

#collecting 25 columns of data in one columns\
data.shape[0]
headlines = []
for row in range(0,data.shape[0]):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
    
    
headlines[0]



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

#Bag of words
#TFiDF
cv=TfidfVectorizer(ngram_range=(2,2))
#cv=CountVectorizer(ngram_range=(2,2))
traindataset = cv.fit_transform(headlines)

# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
#randomclassifier.fit(traindataset,ytrain)
randomclassifier.fit(traindataset,train['Label'])



#predict the test data-----------------------------
## Predict for the Test Dataset
Xtest.shape
test_transform= []
'''
for row in range(0,Xtest.shape[0]):
    test_transform.append(' '.join(str(x) for x in Xtest.iloc[row,1:27]))
'''  
test.shape
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))

test_dataset = cv.transform(test_transform)


#prediction
predictions = randomclassifier.predict(test_dataset)


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


confusion_matrix(test['Label'], predictions)
print(classification_report(test['Label'],predictions))
#54
#--------------------------------------------------

from sklearn.naive_bayes import MultinomialNB
NLP_model = MultinomialNB().fit(traindataset,train['Label'])

y_pred = NLP_model.predict(test_dataset)


m = confusion_matrix(test['Label'], y_pred)
m
from sklearn.metrics import classification_report
print(classification_report(test['Label'],y_pred))







