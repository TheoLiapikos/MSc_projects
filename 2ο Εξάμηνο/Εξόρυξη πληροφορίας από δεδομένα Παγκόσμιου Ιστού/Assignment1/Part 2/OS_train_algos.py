import os
import pandas as pd
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


# Files and Directories used in this part
from files_folders_paths import (path_to_os_trained_classifiers_dir,
                                 path_to_kaggle_preproc_train_data,
                                 path_to_os_trained_clfs_accuracies_df,
                                 path_to_preproc_tweets,
                                 path_to_os_tweets_predictions,
                                 path_to_os_trained_vectorizer)


# Rejoins tokens into string
def join_doc(tokens):
        return(' '.join(tokens))




#########  Choose what to do with Algorithms: ##########
#   mode=1: Train model with fixed (optimized) hyperparameters
#   mode=2: Use Grid Search to optimized hyperparameters
#   mode=3: Use trained algorithms to predict Sentiment on tweets

mode = 1

if(mode == 1): print('\nTrain model with fixed (optimized) hyperparameters\n')
if(mode == 2): print('\nUse Grid Search to optimized hyperparameters\n')
if(mode == 3): print('\nPredicting tweets\' Overall Sentiment\n')



# If directory with Trained Classifiers files doesn't exists create it
if not os.path.exists(path_to_os_trained_classifiers_dir):
    try:  
        os.mkdir(path_to_os_trained_classifiers_dir)
    except OSError:  
        print ('Creation of the directory %s failed.' % path_to_os_trained_classifiers_dir)
    else:  
        print ('Successfully created the directory %s' % path_to_os_trained_classifiers_dir)
else:
    print('Directory %s already exists.' % path_to_os_trained_classifiers_dir)


if(mode == 1 or mode == 2):
    # Load preprocessed training dataset
#    open_file = open(path_to_kaggle_preproc_train_data,"rb")
#    train_df = pickle.load(open_file)
#    open_file.close()
    
    with open(path_to_kaggle_preproc_train_data, "rb") as fp:
       train_df = pickle.load(fp)
    
    # Label distribution of preprocessed training data
    # O: ~34.500   1: ~43.500
    print('\nLabel distribution of preprocessed training data:')
    print(train_df.Sentiment.value_counts())
    
    
    # ******************* FEATURE EXTRACTION
    
    ## 1. Choose a part of training data.
    if(mode == 1):
        # Use all train dataset to train optimized algorithms
        # Dataframe with tweets to process. Drop column with original tweets' text
        data_set = train_df.drop('text', axis=1)
    if(mode == 2):
        # 78.000 tweets is extremely high number to use in GridSearch. I will use part
        # of train dataset. Tweets of each category to choose
        num_tweets = 10000
        
        print('\nSelecting %d tweets of each category from original dataset' %num_tweets)
        
        # Tweets' indices of training dataset having tag 0
        tw0 = list(train_df.index[train_df.Sentiment == 0])
        # Tweets' indices of training dataset having tag 1
        tw1 = list(train_df.index[train_df.Sentiment == 1])
        
        # Randomly choose (wo replacement) from each category the indices of tweets to keep
        keep_tw0 = np.random.choice(tw0, size=num_tweets, replace=False)
        keep_tw1 = np.random.choice(tw1, size=num_tweets, replace=False)
        
        # Dataframe with tweets to process. Drop column with original tweets' text
        data_set = train_df.loc[keep_tw0, :].append(train_df.loc[keep_tw1, :]).drop('text', axis=1)
        # Reset index
        data_set.reset_index(inplace=True, drop=True)
    
        # Label distribution of selected part of training data
        print('\nLabel distribution of selected part of training data:')
        print(data_set.Sentiment.value_counts())
    
    
    # In order to use vectorizer I have to rejoin tokens into strings
    docs = data_set.preprocessed_text.map(join_doc)
    
    # Replace token column with appropriate stings
    data_set['preprocessed_text'] = docs
    
    # Suffle data before splitting variables
    for i in range(5):
        data_set = data_set.iloc[np.random.permutation(np.arange(len(data_set)))].reset_index(drop=True)
    
    # Separate dependent and independent variables
    X = data_set.preprocessed_text
    y = data_set.Sentiment
    
    # Split variables into train (80%) and test (20%) sets
    split_point = int(0.8*len(X))
    
    train_X = X.iloc[:split_point]
    train_y = y.iloc[:split_point]
    
    test_X = X.iloc[split_point:]
    test_y = y.iloc[split_point:]
    
    # Vectorize text. Use distribution of each word. No need to use IDF
    vectorizer = CountVectorizer(ngram_range=(1,1), max_df=0.45)
    
    
    # Fit vectorizer on train set and then transform both train and test sets
    train_X_v = vectorizer.fit_transform(train_X)
    test_X_v = vectorizer.transform(test_X)
    
    # Save vectorizer to use it with predictions
    pickle.dump(vectorizer, open(path_to_os_trained_vectorizer, "wb"))

    # Delete unnecessary variables
    del train_df, data_set, X, y, docs
    if(mode == 2):
        del tw0, tw1, keep_tw0, keep_tw1


#########  Train Classification Algorithms using optimized Hyperparameters values 
if(mode == 1):
    
    # Classification Algorithms to use
    classifiers = [
            ('DecisionTree', DecisionTreeClassifier(criterion='entropy',min_samples_leaf=10,max_features=None,random_state=42)),
            ('MultiNaiveBayes', MultinomialNB(alpha=1.0)),
            ('BernoulliNaiveBayes', BernoulliNB(alpha=1.0)),
            ('LogisticRegression', LogisticRegression(C=0.14,solver='lbfgs',multi_class='multinomial')),
            ('SGD', SGDClassifier(loss='log',penalty='l2',alpha=0.0001,random_state=42)),
    #        ('SVC', SVC()),
            ('LinearSVC', LinearSVC(C=0.75,penalty='l2',loss='hinge',random_state=42)),
    #        ('NuSVC', NuSVC())
            ]
    
    # DataFrame to hold total results
    columns = ['Classifier', 'Accuracy']
    classification_results = pd.DataFrame(columns = columns)
    
    # For each algorithm
    for name,clf_choice in classifiers:
        clf_results = [name, ]
        clf_trained = clf_choice.fit(train_X_v, train_y)
        clf_preds = clf_trained.predict(test_X_v)
        # Classifier's accuracy
        acc = accuracy_score(test_y, clf_preds)
        clf_results.append(acc)

        # Copy results to total DataFrame
        new_row  = pd.Series(clf_results, index= columns)
        classification_results = classification_results.append(new_row, ignore_index=1)
        # Save trained classifier
        path_to_classifier =  os.path.join(path_to_os_trained_classifiers_dir, name)
        # Each classifier is saved as a tuple (name, _trained_classifier)
        save_file = open(path_to_classifier,"wb")
        pickle.dump((name,clf_trained), save_file)
        save_file.close()
    
    # Save total results DataFrame to disc
    save_file = open(path_to_os_trained_clfs_accuracies_df,"wb")
    pickle.dump(classification_results, save_file)
    save_file.close()

   
    ######  Use this code to load results DataFrame to add new Algorithms
#    open_file = open(path_to_os_trained_clfs_accuracies_df,"rb")
#    classification_results = pickle.load(open_file)
#    open_file.close()



#########  Search Grid to set the Hyperparameters values for Classification Algorithms
if(mode == 2):
    
    ##########  Logistic regression
    log_param_grid = [
            {'C':[0.14],
             'solver':['lbfgs'], 'multi_class':['multinomial']
                    }
            ]
    log_clf = GridSearchCV(LogisticRegression(), log_param_grid, cv=3, scoring='accuracy')
    log_clf.fit(train_X, train_y)
    log_predictions = log_clf.predict(test_X)
    print('Optimal parameters for LR: %s' %str(log_clf.best_params_))
    # Optimal parameters for LR: {'C': 0.14, 'multi_class': 'multinomial', 'solver': 'lbfgs'}
    print('Logistic regression Accuracy: %f' % accuracy_score(test_y, log_predictions))
    # Logistic regression Accuracy: 0.710750

    
    ##########  Decision Tree
    tree_param_grid = [
            {'criterion':['entropy'], 'min_samples_leaf':[10],
             'max_features':[None]
                    }
            ]
    tree_clf = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv=3, scoring='accuracy')
    tree_clf.fit(train_X, train_y)
    tree_predictions = tree_clf.predict(test_X)
    print('Optimal parameters for DT: %s' %str(tree_clf.best_params_))
    # Optimal parameters for DT: {'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 75}
    # Optimal parameters for DT: {'criterion': 'gini', 'max_features': None, 'min_samples_leaf': 50}
    # Optimal parameters for DT: {'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 10}
    # Optimal parameters for DT: {'criterion': 'entropy', 'max_features': None, 'min_samples_leaf': 10}
    print('Decision tree Accuracy: %f' % accuracy_score(test_y, tree_predictions))
    # Decision tree Accuracy: 0.642250
    # Decision tree Accuracy: 0.651750
    # Decision tree Accuracy: 0.678750
    # Decision tree Accuracy: 0.678750
    
    
    ##########  Multinomial Naive Bayes
    mnb_param_grid = [
            {'alpha':[1.0]
                    }
            ]
    mnb_clf = GridSearchCV(MultinomialNB(), mnb_param_grid, cv=5, scoring='accuracy')
    mnb_clf.fit(train_X, train_y)
    mnb_predictions = mnb_clf.predict(test_X)
    print('Optimal parameters for MNB: %s' %str(mnb_clf.best_params_))
    # Optimal parameters for MNB: {'alpha': 1}
    print('Multinomial NB Accuracy: %f' % accuracy_score(test_y, mnb_predictions))
    # Multinomial NB Accuracy: 0.706250

    
    ##########  Bernoulli Naive Bayes
    bnb_param_grid = [
            {'alpha':[1.0]
                    }
            ]
    bnb_clf = GridSearchCV(BernoulliNB(), bnb_param_grid, cv=5, scoring='accuracy')
    bnb_clf.fit(train_X, train_y)
    bnb_predictions = bnb_clf.predict(test_X)
    print('Optimal parameters for BNB: %s' %str(bnb_clf.best_params_))
    # )ptimal parameters for BNB: {'alpha': 1}
    print('Bernoulli NB Accuracy: %f' % accuracy_score(test_y, bnb_predictions))
    # Bernoulli NB Accuracy: 0.711500

    
    ##########  SGDClassifier
    sgd_param_grid = [
            {'loss':['log'],
             'penalty':['l2'],
             'alpha':[0.0001],
             'random_state':[42]
                    }
            ]
    sgd_clf = GridSearchCV(SGDClassifier(), sgd_param_grid, cv=3, scoring='accuracy')
    sgd_clf.fit(train_X, train_y)
    sgd_predictions = sgd_clf.predict(test_X)
    print('Optimal parameters for SGD: %s' %str(sgd_clf.best_params_))
    # Optimal parameters for SGD: {'alpha': 0.0001, 'loss': 'log', 'penalty': 'l2'}
    print('SGDClassifier Accuracy: %f' % accuracy_score(test_y, sgd_predictions))
    # SGDClassifier Accuracy: 0.700250

    
    ##########  SVC
    svc_param_grid = [
            {'C':[0.75,1.0,1.25],
             'kernel':['linear','poly','rbf','sigmoid'],
             'random_state':[42]
                    }
            ]
    svc_clf = GridSearchCV(SVC(), svc_param_grid, cv=3, scoring='accuracy')
    svc_clf.fit(train_X, train_y)
    svc_predictions = svc_clf.predict(test_X)
    print('Optimal parameters for SVC: %s' %str(svc_clf.best_params_))
    # Optimal parameters for SVC: {'C': 0.75, 'kernel': 'linear', 'random_state': 42}
    print('SVC Accuracy: %f' % accuracy_score(test_y, svc_predictions))
    # SVC Accuracy: 0.711250

    
    ##########  LinearSVC
    lsvc_param_grid = [
            {'C':[0.75],
             'penalty':['l2'],
             'loss':['hinge'],
             'random_state':[42]
                    }
            ]
    lsvc_clf = GridSearchCV(LinearSVC(), lsvc_param_grid, cv=3, scoring='accuracy')
    lsvc_clf.fit(train_X, train_y)
    lsvc_predictions = lsvc_clf.predict(test_X)
    print('Optimal parameters for LinearSVC: %s' %str(lsvc_clf.best_params_))
    # Optimal parameters for LinearSVC: {'C': 0.75, 'loss': 'hinge', 'penalty': 'l2', 'random_state': 42}
    # Optimal parameters for LinearSVC: {'C': 0.25, 'loss': 'hinge', 'penalty': 'l2', 'random_state': 42}
    print('LinearSVC Accuracy: %f' % accuracy_score(test_y, lsvc_predictions))
    # LinearSVC Accuracy: 0.711000
    # LinearSVC Accuracy: 0.714500

    
    ##########  NuSVC
    nusvc_param_grid = [
            {'nu':[0.25,0.5,0.75],
             'kernel':['linear','poly','rbf','sigmoid'],
             'random_state':[42]
                    }
            ]
    nusvc_clf = GridSearchCV(NuSVC(), nusvc_param_grid, cv=3, scoring='accuracy')
    nusvc_clf.fit(train_X, train_y)
    nusvc_predictions = nusvc_clf.predict(test_X)
    print('Optimal parameters for NuSVC: %s' %str(nusvc_clf.best_params_))
    # 
    print('NuSVC Accuracy: %f' % accuracy_score(test_y, nusvc_predictions))
    # 



#########  Use trained algorithms to predict Sentiment on tweets
if(mode == 3):
    
    from os import listdir
    from os.path import isfile, join
    mypath = path_to_os_trained_classifiers_dir
    clfs_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    # Load trained classifiers
    trained_clfs = []
    for clf in clfs_files:
        path_to_clf = os.path.join(mypath,clf)
        open_file = open(path_to_clf,"rb")
        # Each file contains a tuple (name, trained cld)
        clf_name, trained_clf = pickle.load(open_file)
        open_file.close()
        # Append classifier to list
        trained_clfs.append((clf_name, trained_clf))
    
    # Print existing trained classifiers names
    print('Trained classifiers available for predictions:')
    for i,item in enumerate(trained_clfs):
        print('\t%d: %s'%(i,item[0]))
    
    # Load preprocessed tweets. I only need original and preprocessed text
    open_file = open(path_to_preproc_tweets,"rb")
    tweets_to_pred = pickle.load(open_file)[['text', 'text_preprocessed']]
    open_file.close()
    
    
    # First rejoin preprocessed tweets into stings and the vectorize with vectorizer
    # used on train data,
    tweets_to_pred['text_preprocessed'] = tweets_to_pred['text_preprocessed'].map(join_doc)
    
    # Load stored vectorizer and transform tweets
    vectorizer = pickle.load(open(path_to_os_trained_vectorizer, "rb"))
    tweets_v = vectorizer.transform(tweets_to_pred.text_preprocessed)
    
    ## Make predictions
    cur= pd.DataFrame()
    for name,tr_clf in trained_clfs:
        print(name,tr_clf)
        cur[name] = tr_clf.predict(tweets_v)

    # Voting procedure
    cur['mean'] = cur.mean(axis=1)
    cur['vote'] = 0
    cur['vote'][cur['mean'] > 0.5] = 1
    
    
    tweets_to_pred = pd.concat([tweets_to_pred, cur.drop('mean', axis=1)], axis = 1)
    
    # Save predictions
    save_file = open(path_to_os_tweets_predictions,"wb")
    pickle.dump(tweets_to_pred, save_file)
    save_file.close()

    del cur #, tweets_to_pred

