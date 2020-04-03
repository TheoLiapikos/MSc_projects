from time import time
from pymongo import MongoClient
import pandas as pd

# Functionality to get Unique and Original (no ReTweets) tweets' indices from a collection
from get_original_unique_tweets import get_orig_uniq_idx_from_col

# Files and Directories used in this part
from files_folders_paths import path_to_raw_tweets


# Requirent fields separated by level
root_level_fields = ['created_at','coordinates','text','retweet_count','favorite_count']
user_level_fields = ['id_str','screen_name','followers_count','favourites_count',
                     'statuses_count','listed_count','friends_count','location',
                     'time_zone','utc_offset']
entities_level_fields = ['hashtags','urls','user_mentions']
place_level_fields = ['country_code','full_name','coordinates']


def get_columns_names():
    # Create list of columns' name from tweet's fields
    columns = []
    columns = root_level_fields.copy()
    for field in user_level_fields:
        columns.append('user_'+field)
    for field in entities_level_fields:
        columns.append('entities_'+field)
    for field in place_level_fields:
        columns.append('place_'+field)
    return(columns)


# Basic procedure to Extract and Save in a DataFrame specific tweets (based on
# their indices) from a specific mongoDB collection. Only specific fields from
# each tweet are extracted
def extract_tweets(client,DBname,COLname,indices,columns):
    # Dataframe to hold required fields of original tweets
    tweets_df = pd.DataFrame(columns = columns)
    # Total tweets
    tweets = client[DBname][COLname].find()
    # For each index of original tweets
    for i in indices:
        tweet = tweets[i]
        # Temporary list to hold tweet's fields
        twl = []
        # First collect root-level fields
        for field in root_level_fields:
            twl.append(tweet[field])
        # Then collect the user-level fields
        for field in user_level_fields:
            twl.append(tweet['user'][field])
         # Then collect the entities-level fields
        for field in entities_level_fields:
            twl.append(tweet['entities'][field])
         # Then collect the place-level fields
        for field in place_level_fields:
            try:
                twl.append(tweet['place'][field])
            except:
                twl.append(None)
        twl = pd.Series(twl, index=columns)
        tweets_df = tweets_df.append(twl, ignore_index=1)
    return(tweets_df)


# For large ammount of extracted tweets, use this method that divides the total
# tweets in smaller parts (up to 5000 tweets), accelerating the whole procedure
# Can take advantage of already extracted and saved tweets in a .pkl file (previous_file)
# BEWARE: Imported indices must contain the whole tweets' indices (as extracted above),
#   containing some potentially already existing (previous_file) tweets' indices
#   (starting from index 0), and not only the new tweets to be extracted
# Requires about 48 min for aboout 43.000 tweets, 21 fields for each one
def extract_tweets_in_parts(client,DBname,COLname,indices,columns,previous_file,tws_per_part=5000):

    # Index of already extracted tweets
    previous_tweets = previous_file
    origin = len(previous_tweets)
    print('Already extracted and stored %d tweets' %origin)
    if(len(indices)<=origin):
        print('There are no new tweets to add')
        return(previous_tweets)
    # Indices of new tweets to be extracted
    indices = indices[origin:]
    print('Start extracting %d new tweets...' %len(indices))
    
    t0 = time()
    # The whole procedure divided to smaller parts
    parts = len(indices)//tws_per_part
    for i in range(parts):
        t1 = time()
        temp_tweets = extract_tweets(client,DBname,COLname,indices[i*tws_per_part:(i+1)*tws_per_part],columns)
        previous_tweets = previous_tweets.append(temp_tweets, ignore_index=1)
        t2 = time()
        time_till_now = t2-t0
        mins = time_till_now // 60
        secs = time_till_now % 60
        print('%d tweets extracted in %0d:%02d (%d sec interval)' %((i+1)*tws_per_part, mins, secs, t2-t1))
    
    temp_tweets = extract_tweets(client,DBname,COLname,indices[parts*tws_per_part:],columns)
    tf = time()
    time_till_now = tf-t0
    mins = time_till_now // 60
    secs = time_till_now % 60
    print('Total %d new tweets added in %d:%02d' %(len(indices), mins, secs))
    previous_tweets = previous_tweets.append(temp_tweets, ignore_index=1)
    print('%d tweets extracted and stored in total' %len(previous_tweets))

    return(previous_tweets)



if __name__ == "__main__":
    # Create list of columns' name from tweet's fields
    columns = get_columns_names()

    # First check if there is a file in unproc_tweets' folder with already extracted tweets
    try:
        prev_extr_tweets = pd.read_pickle(path_to_raw_tweets)
    except: # Or else create a new empty file 
        prev_extr_tweets = pd.DataFrame(columns = columns)

    client = MongoClient("localhost", 27017)
    DBname = 'Twitter_Assignment'
    #COLname = "Crypto_Raw_Tweets"
    COLname = 'Clima_Raw_Tweets'
    
    # Get Unique and Original (no ReTweets) tweets' indices from the collection
    print('\n*** Gettting Unique and Original (no ReTweets) tweets\' indices from the collection %s ***' %COLname)
    original_indices = get_orig_uniq_idx_from_col(DBname, COLname)
    
    # Extract Required tweets
    print('\n*** Extracting Unique and Original (no ReTweets) tweets from the collection %s ***' %COLname)
    twdf = extract_tweets_in_parts(client,DBname,COLname,original_indices,columns,prev_extr_tweets)
    
    # Save total Extracted tweets to disc
    print('\n*** Saving extracted tweets from the collection %s ***' %COLname)
    twdf.to_pickle(path_to_raw_tweets)
    print('Total extracted tweets stored on disc under the name \'%s\'' %'raw_tweets.pkl')

