# -*- coding: utf-8 -*-
"""
Created on Thu May 24 20:23:50 2018

@author: HENA
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting=3)
#clean text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review = review.lower() # To make data in lower case
    review=review.split()
    #stemming
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
##MAIN PART
#creating bag of words model through tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer(max_features=1500)#1500 max features used to reduce sparsity which makes better correlation
x=cv.fit_transform(corpus).toarray()#sparse matrix#independent varaiable
y=dataset.iloc[:, 1].values #dependent values, 0 or 1

#Splitting the dataset into training and test dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(x,y,test_size=0.20,random_state=0)

#fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the test result
y_pred=classifier.predict(X_test)

#Make confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)




