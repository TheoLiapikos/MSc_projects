#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:28:27 2019

@author: theo
"""

import pandas as pd
import re

from sklearn import model_selection, metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Load data from .csv file to a DataFrame
data = pd.read_csv('hate_tweets.csv')

###########################   Data PreProcessing   ###########################
# Split text of each tweet to tokens and save tokenized tweets to list
tweets = []
for i in range(len(data)):
    tweet = data["tweet"][i]
    tweets.append([tok for tok in tweet.split(' ')])

# List for preprocessed tweets
clean_tweets = []

# For each tweet (list of tokens) do the following:
#   - Lower the characters
#   - Remove all tokens starting with @
#   - Remove all non-alpharithmetic characters
#   - Remove all non-alphabetical tokens
#   - Remove all tokens with length < 3
#   - (Optional) Stem or Lemmatize tokens
#   - Join resulting tokens to reconstruct tweets
#   - Save preprocessed tweets to new list
ls = LancasterStemmer()
wnl = WordNetLemmatizer()
for tweet in tweets:
    cur = []
    for word in tweet:
        word = word.lower()
        if(word.startswith('@')):
            continue
        word = re.sub(r'[^a-z0-9]+', '',word)
        if(word.isalpha() and len(word)>2):
            cur.append(word)
    # USE ONLY ONE OF THE FOLLOWING OPTIONS TO STORE PREPROCESSED DATA
    # Use this line if you don't want to stem or lemmatize tokens
    clean_tweets.append(' '.join(item for item in cur))
    # (Optional) Use this line instead to stem tokens
#    clean_tweets.append(' '.join(ls.stem(item) for item in cur))
    # (Optional) Use this line instead to lemmatize tokens
#    clean_tweets.append(' '.join(wnl.lemmatize(item) for item in cur))
 
# Add preprocessed tweets to DataFrame
data['clean_tweets'] = clean_tweets

# Defining X and y data
X = list(data['clean_tweets'])
y = list(data['class'])

# Split data to Train and Test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,stratify=y,random_state = 42)


# Method to evaluate and print results returning from GridSearch
def evalBestPipe(gridSearchResults):
    # Extract best parameters
    best_pipe = gridSearchResults.best_estimator_

    # Best Grid results
    gsResults = gridSearchResults.cv_results_

    # Grid validation error
    meanTestAcc = gsResults['mean_test_score'][gridSearchResults.best_index_]
    stdTestAcc = gsResults['std_test_score'][gridSearchResults.best_index_]

    # Grid training error
    meanTrainAcc = gsResults['mean_train_score'][gridSearchResults.best_index_]
    stdTrainAcc = gsResults['std_train_score'][gridSearchResults.best_index_]
    
    # Predictions over test data
    y_pred = best_pipe.predict(X_test)
    
    # Classification report (returns all usefull metrics)
    classificationReport = metrics.classification_report(y_test, y_pred)
    print("\nClassification Report: \n", classificationReport)
    
    # Print basic metrics from Grid
    print("\nTraining Accuracy: %0.3f (+/- %0.2f)" % (meanTrainAcc, stdTrainAcc))
    print("Validation Accuracy: %0.3f (+/- %0.2f)" % (meanTestAcc, stdTestAcc))
    print("\nΒέλτιστες παράμετροι: ", gridSearchResults.best_params_)
    


###################   Multinomial Naive Bayes Algorithm   ####################
# I use TF-IDF method to vectorize tweets. I combine vectorizer and classifier
# in a pipeline
multiNB_pipe = Pipeline([
    ('TfIdf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Best values for basic parameters of vectorizer and classifier. These are
# the result of extensive research of all possible values' range using
# GridSearchCV() method.
params_NB = {
    'TfIdf__max_features': (10000,),
    'TfIdf__max_df': (0.35,),
    'TfIdf__min_df': (5,),
    'TfIdf__ngram_range': ((1,3),),
    'TfIdf__use_idf': (True,),
    'TfIdf__norm': ('l2',),
    'clf__alpha': (0.01,),
    'clf__fit_prior': (True,),
}

# GridSearc method to oprimize algorithms' parameters
grid_search_NB = GridSearchCV(multiNB_pipe, params_NB, cv=10, n_jobs=3,
                              verbose=1, return_train_score=True)

# Fit model on train data
grid_search_NB.fit(X_train, y_train)

# Evaluate model 
print('\n\n*********** Multinomial Naive Bayes Algorithm ***********')
evalBestPipe(grid_search_NB)



#####################   Logistic Regression Algorithm   ######################
# I use TF-IDF method to vectorize tweets. I combine vectorizer and classifier
# in a pipeline
LogReg_pipe = Pipeline([
    ('TfIdf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Best values for basic parameters of vectorizer and classifier. These are
# the result of extensive research of all possible values' range using
# GridSearchCV() method.
params_LR = {
    'TfIdf__max_features': (10000,),
    'TfIdf__max_df': (0.45,),
    'TfIdf__min_df': (5,),
    'TfIdf__ngram_range': ((1,2),),
    'TfIdf__use_idf': (True,),
    'TfIdf__norm': ('l2',),
    'clf__C': (10,),
    'clf__solver': ('saga',),
    'clf__multi_class': ('multinomial',),
}

# GridSearc method to oprimize algorithms' parameters
grid_search_LR = GridSearchCV(LogReg_pipe, params_LR, cv=10, n_jobs=3,
                              verbose=1, return_train_score=True)

# Fit model on train data
grid_search_LR.fit(X_train, y_train)

# Evaluate model
print('\n\n*********** Logistic Regression Algorithm ***********')
evalBestPipe(grid_search_LR)




#######################   Support Vector Machines   #########################
# I use TF-IDF method to vectorize tweets. I combine vectorizer and classifier
# in a pipeline
SVC_pipe = Pipeline([
    ('TfIdf', TfidfVectorizer()),
    ('clf', SVC())
])

# Best values for basic parameters of vectorizer and classifier. These are
# the result of extensive research of all possible values' range using
# GridSearchCV() method.
params_SVC = {
    'TfIdf__max_features': (5000,),
    'TfIdf__max_df': (0.35,),
    'TfIdf__min_df': (3,),
    'TfIdf__ngram_range': ((1,2),),
    'TfIdf__use_idf': (True,),
    'TfIdf__norm': ('l2',),
    'clf__C': (1,),
    'clf__kernel': ('linear',),
    'clf__degree': (1,),
}

# GridSearc method to oprimize algorithms' parameters
grid_search_SVC = GridSearchCV(SVC_pipe, params_SVC, cv=10, n_jobs=2,
                              verbose=1, return_train_score=True)

# Fit model on train data
grid_search_SVC.fit(X_train, y_train)

# Evaluate model
print('\n\n*********** Support Vector Machines Algorithm ***********')
evalBestPipe(grid_search_SVC)




















