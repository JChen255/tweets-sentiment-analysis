# Brief Summary of Project

This project trained several models using different training datasets with Logistic Regression and Bernouilli Naive Bayes models to fulfill the sentiment analysis task. 

Five training datasets were used to train the classification model, including [sentiment 140], [Apple Twitter Sentiment], [Twitter US Airline Sentime], [Depression Sentiment], and [Russia invade tweets]. These models generated were then tested on the [Putin tweets] dataset to demonstrate their accuracy in predicting tweet content related to Russian president Putin. ([Russia invade tweets] and [Putin tweets] are attached in this file) 

[sentiment 140]: https://www.kaggle.com/datasets/kazanova/sentiment140
[Apple Twitter Sentiment]: https://data.world/crowdflower/apple-twitter-sentiment/workspace/file?filename=Apple-Twitter-Sentiment-DFE.csv
[Twitter US Airline Sentime]: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
[Depression Sentiment]: https://www.kaggle.com/code/tarunkumar120/twitter-tweet-sentiment-analysis-96-accuracy


# How to use the codes? / usage example

Introduction: There are five .py files: preprocess.py, building_model.py, evaluatemodel.py, predicting.py, and analyzing.py. The preprocess.py and evaluate.py are two helper files for builing_model.py and predicting.py. Finally, the analyzing.py is for analyzing our dataset.
building_model.py:

Firstly, you should import the dataset. Then, you should choose different commands and modify the parameters following the comments based on the dataset you upload. After that, the code will preprocess the data, split it into train and test datasets, and transform X_train into tf-idf features. Afterward, the code will create and evaluate a Bernoulli Naive Bayes model and a Logistic Regression model. Finally, you can save the vectorizer and models into pickle files.

How to use the [predicting_model.py]:
First, download the vectorizer and models from pickle files. Second, download the text and labels of the test dataset. Third, use the models to make predictions. Fourth, calculate the specificity scores and metrics.

How to use the [analyzing.py]:
The file has two functions. Firstly, it can create the wordnet plot and list out the top negative and non-negative words in a few datasets. Secondly, it can label the dataset using VADER models.
