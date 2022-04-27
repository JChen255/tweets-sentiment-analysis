# utilities
import re
import pickle
import numpy as np
import pandas as pd
import random

# plotting
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# nltk
from nltk.stem import WordNetLemmatizer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# helper function
from preprocess import preprocess
from evaluatemodel import model_Evaluate

# Importing the dataset(Importing Different Dataset)
#DATASET_COLUMNS  = ["Index", "message to examine", "label (depression result)"]
#DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv('Apple_Twitter_traindata.csv')

# Removing the unnecessary columns.
dataset = dataset[['text','label']]

# Shuffle the dataset
dataset.sample(frac=1.0)
# Replacing the values to ease understanding.
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

# Storing data in lists.
text, sentiment = list(dataset['text']), list(dataset['label'])
processedtext = preprocess(text)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment,
                                                    test_size = 0.05, random_state = 0)
print(f'Data Split done.')

# TF-IDF Vectoriser
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

# Transforming the data
X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')

# Creating and Evaluating Models
# BernoulliNB Model
BNBmodel = BernoulliNB(alpha = 2)
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel,X_test,y_test)

#Logistic Regression Model
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel,X_test,y_test)

#Saving the Models
file = open('vectoriser-ngram-(1,2)_apple.pickle','wb')
pickle.dump(vectoriser, file)
file.close()

file = open('Sentiment-LR_apple.pickle','wb')
pickle.dump(LRmodel, file)
file.close()

file = open('Sentiment-BNB_apple.pickle','wb')
pickle.dump(BNBmodel, file)
file.close()