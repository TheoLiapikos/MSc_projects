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
from files_folders_paths import (path_to_spem_trained_classifiers_dir,
                                path_to_spem_preproc_anger_train_data,
                                path_to_spem_preproc_disgust_train_data,
                                path_to_spem_preproc_fear_train_data,
                                path_to_spem_preproc_joy_train_data,
                                path_to_spem_preproc_sadness_train_data,
                                path_to_spem_preproc_surprise_train_data,
                                path_to_spem_preproc_rebal_anger_train_data,
                                path_to_spem_preproc_rebal_disgust_train_data,
                                path_to_spem_preproc_rebal_fear_train_data,
                                path_to_spem_preproc_rebal_joy_train_data,
                                path_to_spem_preproc_rebal_sadness_train_data,
                                path_to_spem_preproc_rebal_surprise_train_data,
                                path_to_spem_prepoc_trained_clfs_accuracies_df,
                                path_to_spem_prepoc_rebal_trained_clfs_accuracies_df,
                                path_to_preproc_tweets,
                                path_to_spem_preproc_tweets_predictions,
                                path_to_spem_preproc_rebal_tweets_predictions
                                 )


# Rejoins tokens into string
def join_doc(tokens):
        return(' '.join(tokens))


#########  Choose what to do with Algorithms: ##########
#   mode=1: Train model with preprocessed datasets (fixed (optimized) hyperparameters)
#   mode=2: Train model with preprocessed rebalanced datasets (fixed (optimized) hyperparameters)
#   mode=3: Use trained algorithms to predict Sentiment on tweets (trained on preprocessed datasets)
#   mode=4: Use trained algorithms to predict Sentiment on tweets (trained on preprocessed rebalanced datasets)

mode = 2


if(mode == 1): print('\nTrain model with preprocessed datasets (fixed (optimized) hyperparameters)\n')
if(mode == 2): print('\nTrain model with preprocessed rebalanced datasets (fixed (optimized) hyperparameters)\n')
if(mode == 3): print('\nUse trained algorithms to predict Sentiment on tweets (trained on preprocessed datasets)\n')
if(mode == 4): print('\nUse trained algorithms to predict Sentiment on tweets (trained on preprocessed rebalanced datasets)\n')


# If directory with Trained Classifiers files doesn't exists create it
if not os.path.exists(path_to_spem_trained_classifiers_dir):
    try:  
        os.mkdir(path_to_spem_trained_classifiers_dir)
    except OSError:  
        print ('Creation of the directory %s failed.' % path_to_spem_trained_classifiers_dir)
    else:  
        print ('Successfully created the directory %s' % path_to_spem_trained_classifiers_dir)
else:
    print('Directory %s already exists.' % path_to_spem_trained_classifiers_dir)



if(mode == 1 or mode == 2):
    # If no rebalanced datasets used
    if(mode == 1):
        
        train_datasets = [
                            path_to_spem_preproc_anger_train_data,
                            path_to_spem_preproc_disgust_train_data,
                            path_to_spem_preproc_fear_train_data,
                            path_to_spem_preproc_joy_train_data,
                            path_to_spem_preproc_sadness_train_data,
                            path_to_spem_preproc_surprise_train_data
                        ]
        
    
    # If rebalanced datasets used
    if(mode == 2):
        
        train_datasets = [
                            path_to_spem_preproc_rebal_anger_train_data,
                            path_to_spem_preproc_rebal_disgust_train_data,
                            path_to_spem_preproc_rebal_fear_train_data,
                            path_to_spem_preproc_rebal_joy_train_data,
                            path_to_spem_preproc_rebal_sadness_train_data,
                            path_to_spem_preproc_rebal_surprise_train_data
                        ]
        
    
    # DataFrame to hold total results
    columns = ['Emotion', 'Classifier', 'Accuracy']
    classification_results = pd.DataFrame(columns = columns)
    
    # Open one dataset (emotion) the time
    for load_file in train_datasets:
        # Load the corresponding dataset
        dataset = pickle.load(open(load_file, "rb"))
        # Dataset's emotion
        emotion = dataset.columns[2]
        print('\n\nTraining algorithms on \'%s\' emotion' %emotion)
        
        # Label distribution of preprocessed training data
        print('\nLabel distribution of training data:')
        print(dataset[emotion].value_counts())
    
        
        # ******************* FEATURE EXTRACTION
        
        # In order to use vectorizer I have to rejoin tokens into strings
        docs = dataset.preprocessed_text.map(join_doc)
        
        # Replace token column with appropriate stings
        dataset['preprocessed_text'] = docs
        
        # Suffle data before splitting variables
        for i in range(5):
            dataset = dataset.iloc[np.random.permutation(np.arange(len(dataset)))].reset_index(drop=True)
        
        # Separate dependent and independent variables
        X = dataset.preprocessed_text
        y = dataset[emotion]
        
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
    
        # Delete unnecessary variables
        del dataset,X,y,docs
    
    
        #########  Train Classification Algorithms using optimized Hyperparameters values 
        
        # Classification Algorithms to use
        classifiers = [
                ('DecisionTree', DecisionTreeClassifier(criterion='entropy',min_samples_leaf=10,
                                                        max_features=None,random_state=42)),
                ('MultiNaiveBayes', MultinomialNB(alpha=1.0)),
                ('BernoulliNaiveBayes', BernoulliNB(alpha=1.0)),
                ('LogisticRegression', LogisticRegression(C=0.14,solver='lbfgs',multi_class='multinomial')),
                ('SGD', SGDClassifier(loss='log',penalty='l2',alpha=0.0001,random_state=42)),
        #        ('SVC', SVC()),
                ('LinearSVC', LinearSVC(C=0.75,penalty='l2',loss='hinge',random_state=42)),
        #        ('NuSVC', NuSVC())
                ]
        
        # List of tuples to hold trained classifiers
        trained_clfs = []
        
        # For each algorithm
        for name,clf_choice in classifiers:
            clf_results = [emotion, name, ]
            clf_trained = clf_choice.fit(train_X_v, train_y)
            clf_preds = clf_trained.predict(test_X_v)
            # Classifier's accuracy
            acc = accuracy_score(test_y, clf_preds)
            clf_results.append(acc)
    
            # Copy results to total DataFrame
            new_row  = pd.Series(clf_results, index= columns)
            classification_results = classification_results.append(new_row, ignore_index=1)
            # Keep trained classifier in list
            trained_clfs.append((name,clf_trained))
            
        # Save trained classifier for particular emotion (plus vectorizer)
        if(mode == 1):
            file_name = 'preproc_' + emotion + '_trained_clfs'
        elif(mode == 2):
            file_name = 'preproc_rebal_' + emotion + '_trained_clfs'
        path_to_emo_clfs =  os.path.join(path_to_spem_trained_classifiers_dir, file_name)
        # Each classifier is saved as a tuple (name, trained_classifier, vectorizer)
        pickle.dump((emotion,trained_clfs,vectorizer), open(path_to_emo_clfs, "wb"))
    
    # Save total results DataFrame to disc
    if(mode == 1):
        destination = path_to_spem_prepoc_trained_clfs_accuracies_df
    elif(mode == 2):
        destination = path_to_spem_prepoc_rebal_trained_clfs_accuracies_df
    pickle.dump(classification_results, open(destination, "wb"))
    
    del clf_preds, test_X, test_y, train_X, train_y, trained_clfs


#########  Use trained algorithms to predict Sentiment on tweets
if(mode == 3 or mode == 4):
    
    from os import listdir
    from os.path import isfile, join
    mypath = path_to_spem_trained_classifiers_dir
    clfs_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    # Split classifiers whether they trained on rebalanced dataset or not 
    clfs_preproc_rebal = sorted([clf for clf in clfs_files if clf.startswith('preproc_rebal_')])
    clfs_preproc = sorted([clf for clf in clfs_files if clf not in clfs_preproc_rebal])

    
    # Load trained classifiers according to training dataset used
    preproc_trained_clfs = []
    for clf in clfs_preproc:
        path_to_clf = os.path.join(mypath,clf)
        # Each file contains a tuple (emotion, trained_clfs list, vectorizer)
        emotion, trained_clfs, vectorizer = pickle.load(open(path_to_clf, "rb"))
        # Append classifier to list
        preproc_trained_clfs.append((emotion, trained_clfs, vectorizer))
        
    preproc_rebal_trained_clfs = []
    for clf in clfs_preproc_rebal:
        path_to_clf = os.path.join(mypath,clf)
        # Each file contains a tuple (emotion, trained_clfs list, vectorizer)
        emotion, trained_clfs, vectorizer = pickle.load(open(path_to_clf, "rb"))
        # Append classifier to list
        preproc_rebal_trained_clfs.append((emotion, trained_clfs, vectorizer))
    
    # Load preprocessed tweets. I only need original and preprocessed text
    tweets_to_pred = pickle.load(open(path_to_preproc_tweets, "rb"))[['text', 'text_preprocessed']]
    
    # First rejoin preprocessed tweets into stings and the vectorize with vectorizer
    # used on train data,
    tweets_to_pred['text_preprocessed'] = tweets_to_pred['text_preprocessed'].map(join_doc)
    
    tweets_only_votes = tweets_to_pred.copy()
    
    # Set trained clfs to be used
    if(mode == 3):
        emo_clfs = clfs_preproc
    elif(mode == 4):
        emo_clfs = clfs_preproc_rebal
    
    
    # Make predictions for each individual emotion
    # Load files with trained classifiers
    for tr_clf_filename in emo_clfs:
        file_name = os.path.join(path_to_spem_trained_classifiers_dir, tr_clf_filename)
        # Each member in list contains a tuple (emotion, trained_clfs list, vectorizer)
        (emotion, trained_clfs, vectorizer) = pickle.load(open(file_name, "rb"))
        # Dataframe to hold predictions for each emotion
        cur= pd.DataFrame()
        # Use particular vectorizer to transform tweets' text
        tweets_v = vectorizer.transform(tweets_to_pred.text_preprocessed)
        # Make predictions for each trained classifier
        for name,tr_clf in trained_clfs:
            # Save predictions to current dataframe
            cur[emotion+'_'+name] = tr_clf.predict(tweets_v)
    
        # Voting procedure
        cur['mean'] = cur.mean(axis=1)
        cur[emotion+'_vote'] = 0
        cur[emotion+'_vote'][cur['mean'] > 0.5] = 1
        
        # Update total results
        tweets_to_pred = pd.concat([tweets_to_pred, cur.drop('mean', axis=1)], axis = 1)
        tweets_only_votes = pd.concat([tweets_only_votes, cur[emotion+'_vote']], axis = 1)
    
    # Finally save total DataFrame to disc
    if(mode == 3):
        file_name = path_to_spem_preproc_tweets_predictions
        file_name2 = path_to_spem_preproc_tweets_predictions+'_only_Votes'
    elif(mode == 4):
        file_name = path_to_spem_preproc_rebal_tweets_predictions
        file_name2 = path_to_spem_preproc_rebal_tweets_predictions+'_only_Votes'
    
    pickle.dump(tweets_to_pred, open(file_name, "wb"))
    pickle.dump(tweets_only_votes, open(file_name2, "wb"))

    
    del cur, tweets_to_pred

    # Statistical analysis of results
    print('\n\nStatistical analysis of tweets\' classification:')
    print('\n%d tweets classified in total' % tweets_only_votes.shape[0])
    

    print('\nStatistical analysis (emotion based):')
    for i in range(2,8,1):
        # The column of each emotion
        col = tweets_only_votes.iloc[:, i:i+1]
        # Emotion name
        emotion = col.sum().index[0][:-5]
        # Emotion counts
        counts = col.sum()
        print('\tEmotion label \'%s\' assigned to \t%d tweets  (%.1f%%)'
              %(emotion, counts, 100*counts/tweets_only_votes.shape[0]))

    print('\nStatistical analysis (tweet based):')
    # Add an extra column holding the total number of emotions each tweets classified to
    tweets_only_votes['tweets_#_emos']= tweets_only_votes.iloc[:, 2:8].sum(axis=1)
    distr = tweets_only_votes['tweets_#_emos'].value_counts()
    for i in range(6):
        try:
            print('\t%d tweets classified to \t%d emotions (%.1f%%)'
                  %(distr[i], i, 100*distr[i]/tweets_only_votes.shape[0]))
        except:
            pass


    del col

