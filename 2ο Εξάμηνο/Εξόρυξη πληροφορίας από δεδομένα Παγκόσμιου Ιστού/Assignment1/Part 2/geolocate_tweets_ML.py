from time import time
import os
import pickle
from collections import Counter
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


# Files and Directories used in this part
from files_folders_paths import (path_to_raw_tweets,
                                 path_to_geoloc_dir
                                 )


# Recreate columns names used on tweets
def get_columns_names():
    # Create list of columns' name from tweet's fields
    columns = []
    columns = root_level_fields.copy()
    for field in user_level_fields:
        columns.append('user_'+field)
    for field in place_level_fields:
        columns.append('place_'+field)
    return(columns)


# Helper function to parse time from a datetime stamp and return the time in seconds
def turn_to_seconds(row):
#    time_str = parser.parse(row).strftime('%H:%M:%S')
#    h, m, s = time_str.split(':')
    return int(row[8:10]) * 3600 + int(row[11:13]) * 60 + int(row[14:16])


# Tokenizer to use for text vectorization
def tokenize(tweet):
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tokenizer.tokenize(tweet)



#########  Choose what to do with Algorithms: ##########
#   mode=1: Train Machine Learning models and save them on disc
#   mode=2: Use saved trained models to perform predictions


mode = 1

if(mode == 1): print('\nTrain Machine Learning models\n')
if(mode == 2): print('\nPerform predictions using trained model\n')


# Train the machine Learning model
if(mode == 1):
    t0 = time()
    
    # If directory with Geolocation Analysis files doesn't exists create it
    if not os.path.exists(path_to_geoloc_dir):
        try:  
            os.mkdir(path_to_geoloc_dir)
        except OSError:  
            print ('Creation of the directory %s failed.' % path_to_geoloc_dir)
        else:  
            print ('Successfully created the directory %s' % path_to_geoloc_dir)
    else:
        print('Directory %s already exists.' % path_to_geoloc_dir)
    
    
    #### Feature Selection
    # Raw Tweets's columns to keep as features for following classifications
    # user_utc_offset and place_coordinates are all NONE so dont use them
    root_level_fields = ['text','created_at','coordinates']
    user_level_fields = ['screen_name','followers_count','favourites_count',
                         'statuses_count','listed_count','friends_count','location']
    place_level_fields = ['country_code','full_name']
    
    
    columns = get_columns_names()
    
    # Load raw tweets. Only the fields needed for classification
    df = pickle.load(open(path_to_raw_tweets, "rb"))[columns]
    
    
    #### Modify features type
    # Turn time of tweets' creation into sec. Store new feature in a new column
    # called 'tweet_time_secs'
    df['tweet_time_secs'] = [turn_to_seconds(row) for row in df['created_at']]
    
    
    #### Data Preprocessing
    # Deal with missing data
    # The field 'user_location' is nullable. If the value is missing I replace it
    # with an indicator value
    df.loc[df['user_location'] == '', 'user_location'] = 'Missing_Value' 
    
    
    #### Aggregating Tweets by User
    
    # First, in order to aggregate values, I have to remove no numeric types from
    # fields 'user_friends_count', 'user_favourites_count', 'user_listed_count' and
    # 'user_followers_count'
    df.loc[df['user_friends_count'] == '', 'user_friends_count'] = '' 
    df.loc[df['user_favourites_count'] == '', 'user_favourites_count'] = '' 
    df.loc[df['user_listed_count'] == '', 'user_listed_count'] = '' 
    df.loc[df['user_followers_count'] == '', 'user_followers_count'] = '' 
    
    # Aggregation functions applied on various fields
    agg_funcs = {'text' : lambda x: ' '.join(x),
                 'user_location': lambda x: Counter(x).most_common(1)[0][0],
                 'place_country_code': lambda x: Counter(x).most_common(1)[0][0],
                 'place_full_name' : lambda x: Counter(x).most_common(1)[0][0],
                 'tweet_time_secs' : np.median, 
                 'user_statuses_count': np.max,
                 'user_friends_count' :np.mean,
                 'user_favourites_count' : np.mean,
                 'user_listed_count' : np.mean,
                 'user_followers_count' : np.mean
                 }
                 
                 
    # Groupby 'screen_name' and then apply the aggregation functions defined above
    df = df.groupby(['user_screen_name']).agg(agg_funcs).reset_index()
    
    
    # 1. Prepare the cleaned data for model fitting
    
    # Read cleaned training tweets file into pandas and randomize it
    #df = pd.read_pickle('cleaned_training_tweets.pkl')
    randomized_df = df.sample(frac=1, random_state=42)
    
    # Split randomized_df into two disjoint sets
    half_randomized_df = int(randomized_df.shape[0] / 2)
    base_df = randomized_df.iloc[:half_randomized_df, :]      # used to train the base classifiers
    meta_df = randomized_df.iloc[half_randomized_df:, :]      # used to train the meta classifier
    
    # Create variables for the known the geotagged locations from each set
    base_y = base_df['user_location'].values
    meta_y = meta_df['user_location'].values
    
    
    # 2. Train a base-level Linear SVC classifier on the user described locations
    # Raw text of user described locations
    base_location_doc = base_df['user_location'].values
    meta_location_doc = meta_df['user_location'].values
    
    # Use a TF-IDF vectorizer. Fit_Transform the vectorizer on base_location_doc and
    # then Transform meta_location_doc
    location_vect = TfidfVectorizer(stop_words='english', tokenizer=tokenize, ngram_range=(1,2))
    base_location_X = location_vect.fit_transform(base_location_doc.ravel())
    meta_location_X = location_vect.transform(meta_location_doc)
    
    # Fit a Linear SVC Model with 'base_location_X' and 'base_y'. It is important to use 
    # balanced class weights otherwise the model will overwhelmingly favor the majority class.
    location_SVC = LinearSVC(class_weight='balanced')
    location_SVC.fit(base_location_X, base_y)
    
    # We can now pass meta_location_X into the fitted model and save the decision 
    # function, which will be used in Step 4 when we train the meta random forest
    location_SVC_decsfunc = location_SVC.decision_function(meta_location_X)
    
    # Save the location vectorizer and the trained linear SVC model for future use
    filename = os.path.join(path_to_geoloc_dir, 'User_Location_Vectorizer')
    with open(filename, "wb") as fp:
        pickle.dump(location_vect, fp)
    filename = os.path.join(path_to_geoloc_dir, 'User_Location_Linear_SVC')
    with open(filename, "wb") as fp:
        pickle.dump(location_SVC, fp)
    
    
    # 3. Train a base-level Linear SVC classifier on the tweets
    # Raw text of tweets
    base_tweet_doc = base_df['text'].values
    meta_tweet_doc = meta_df['text'].values
    
    # Use a TF-IDF vectorizer. Fit_Transform the vectorizer on base_tweet_doc and
    # then Transform meta_tweet_doc
    tweet_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=tokenize)
    base_tweet_X = tweet_vectorizer.fit_transform(base_tweet_doc.ravel())
    meta_tweet_X = tweet_vectorizer.transform(meta_tweet_doc)
    
    # Fit a Linear SVC Model with 'base_tweet_X' and 'base_tweet_y'. It is important to use 
    # balanced class weights otherwise the model will overwhelmingly favor the majority class.
    tweet_SVC = LinearSVC(class_weight='balanced')
    tweet_SVC.fit(base_tweet_X, base_y)
    
    # We can now pass meta_tweet_X into the fitted model and save the decision 
    # function, which will be used in Step 4 when we train the meta random forest
    tweet_SVC_decsfunc = tweet_SVC.decision_function(meta_tweet_X)
    
    # Save the tweet vectorizer and the trained linear SVC model for future use
    filename = os.path.join(path_to_geoloc_dir, 'Tweet_Vectorizer')
    with open(filename, "wb") as fp:
        pickle.dump(tweet_vectorizer, fp)
    filename = os.path.join(path_to_geoloc_dir, 'Tweet_Linear_SVC')
    with open(filename, "wb") as fp:
        pickle.dump(tweet_SVC, fp)
    
    
    # 4. Train a meta-level Random Forest classifier
    # Additional features from meta_df to pull into the final model
    friends_count = meta_df['user_friends_count'].values.reshape(meta_df.shape[0], 1)
    tweet_time_secs = meta_df['tweet_time_secs'].values.reshape(meta_df.shape[0], 1)
    statuses_count = meta_df['user_statuses_count'].values.reshape(meta_df.shape[0], 1)
    favourites_count = meta_df['user_favourites_count'].values.reshape(meta_df.shape[0], 1)
    followers_count = meta_df['user_followers_count'].values.reshape(meta_df.shape[0], 1)
    listed_count = meta_df['user_listed_count'].values.reshape(meta_df.shape[0], 1)
    
    # Stack (using np.hstack) these additional features together
    add_features = np.hstack((friends_count, 
                              tweet_time_secs,
                              statuses_count,
                              favourites_count,
                              followers_count,
                              listed_count))
    
    # Stack (using np.hstack) the two decision function variables from steps 2 & 3
    # with added above_features
    meta_X = np.hstack((location_SVC_decsfunc,        # from Step 2 above
                        tweet_SVC_decsfunc,           # from Step 3 above
                        add_features))
    
    # Fit Random Forest with 'meta_X' and 'meta_y'
    # EXTREMELY SLOW PROCEDURE
    meta_RF = RandomForestClassifier(n_estimators=60, n_jobs=-1)
    meta_RF.fit(meta_X, meta_y)
    
    # Save trained Random Forest model for future use
    filename = os.path.join(path_to_geoloc_dir, 'Random_Forest_model')
    with open(filename, "wb") as fp:
        pickle.dump(meta_RF, fp)
    
    
    print('Total time needed: #d sec' %(time()-t0))





# Make predictions using trained models
if(mode == 2):    
    def make_predictions(df):
        # Get 'user_location' column
        location_doc = df['user_location'].values
        
        # Get tweets' text
        tweet_doc = df['text'].values
        
        # Vectorize 'location_doc' and 'tweet_doc'
        location_X = location_vectorizer.transform(location_doc.ravel())
        tweet_X = tweet_vectorizer.transform(tweet_doc.ravel())
        
        # Store decision functions for 'location_X' and 'tweet_X'
        location_decision_function = location_SVC.decision_function(location_X)
        tweet_decision_function = tweet_SVC.decision_function(tweet_X)
        
        # Additional features from meta_df to pull into the final model
        friends_count = df['user_friends_count'].values.reshape(df.shape[0], 1)
        tweet_time_secs = df['tweet_time_secs'].values.reshape(df.shape[0], 1)
        statuses_count = df['user_statuses_count'].values.reshape(df.shape[0], 1)
        favourites_count = df['user_favourites_count'].values.reshape(df.shape[0], 1)
        followers_count = df['user_followers_count'].values.reshape(df.shape[0], 1)
        listed_count = df['user_listed_count'].values.reshape(df.shape[0], 1)
    
        # Stack (using np.hstack) these additional features together
        add_features = np.hstack((friends_count, 
                               tweet_time_secs,
                               statuses_count,
                               favourites_count,
                               followers_count,
                               listed_count))
        
        # Stack (using np.hstack) the two decision function variables with above
        # add_features
        meta_X = np.hstack((location_decision_function, tweet_decision_function, add_features))
    
        # Feed meta_X into Random Forest and make predictions
        return random_forest.predict(meta_X)
    
    
    if __name__ == "__main__":
        
        # Load trained models from disc
        if(1):
            filename = os.path.join(path_to_geoloc_dir, 'User_Location_Vectorizer')
            with open(filename, "rb") as fp:
                location_vectorizer = pickle.load(fp) 
            filename = os.path.join(path_to_geoloc_dir, 'User_Location_Linear_SVC')
            with open(filename, "rb") as fp:
                location_SVC = pickle.load(fp) 
            filename = os.path.join(path_to_geoloc_dir, 'Tweet_Vectorizer')
            with open(filename, "rb") as fp:
                tweet_vectorizer = pickle.load(fp) 
            filename = os.path.join(path_to_geoloc_dir, 'Tweet_Linear_SVC')
            with open(filename, "rb") as fp:
                tweet_SVC = pickle.load(fp) 
            filename = os.path.join(path_to_geoloc_dir, 'Random_Forest_model')
            with open(filename, "rb") as fp:
                random_forest = pickle.load(fp)    
        
        
        # Load a DataFrame holding the Tweets you want to predict. Dataframe should
        # have columns and columns names as those used in the train part
        evaluation_df = pd.read_pickle('evaluation_df.pkl')
        
        # Get predicted locations
        predictions = make_predictions(evaluation_df)
        
        # Create a new column called 'predicted_location'
        evaluation_df.loc[:, 'predicted_location'] = predictions
        
        # Save the resulting DataFrame with the location predictions
        filename = os.path.join(path_to_geoloc_dir, 'Locations_perdictions_ML')
        evaluation_df.to_pickle(filename)
    
    
    
    

