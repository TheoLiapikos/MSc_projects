import pandas as pd
from pymongo import MongoClient

'''
Get the INDICES of UNIQUE and ORIGINAL (no ReTweets) tweets of a specific Collection
using only the indices and the ids to save a lot of time
Execution time less than 1 min for a collection with over 100.000 raw tweets
'''

# Returns a DataFrame with ALL indices and ids of a specific Collection's tweets 
def get_raw_idx_id_df(client,DBname,COLname):
    # ALL Collection's tweets
    raw_tweets = client[DBname][COLname]
    raw_indices = [] # Αυτό μάλλον δεν χρειάζεται και μπορώ να χρησιμοπποιήσω το index του DataFrame
    raw_ids = []
    index = 0
    for tweet in raw_tweets.find():
        raw_indices.append(index)
        raw_ids.append(tweet['id'])
        index+=1
    print('Collection contains %d raw tweets in total'  %len(raw_ids))
    df = pd.DataFrame()
    df['index'] = raw_indices
    df['id'] = raw_ids
    return(df)

# Imports DataFrames (from previous step) and returns LIST of UNIQUE tweets indices
def get_uniq_idx(sorted_raw_df):
    list_uniq_idx = []
    list_uniq_idx.append(sorted_raw_df['index'][0])
    for i in range(1,len(sorted_raw_df)):
        if(sorted_raw_df['id'][i] > sorted_raw_df['id'][i-1]):
            list_uniq_idx.append(sorted_raw_df['index'][i])
    print('Collection contains %d Unique tweets'  %len(list_uniq_idx))
    return(list_uniq_idx)

# Returns a LIST of ORIGINAL (no ReTweets) tweets' indices of a specific Collection 
def get_orig_uniq_idx(client,DBname,COLname):
    retweets = 0
    original = 0
    or_indices = []
    index = 0
    for tweet in client[DBname][COLname].find():
        try:
            if(tweet['retweeted_status']):
                retweets+=1
        except:
            original+=1
            or_indices.append(index)
        index+=1
    print('Collection contains %d Original tweets and %d ReTweets' %(original,retweets))
    return(or_indices)

# Use this function as IMPORT to get full functionality from within an external
# .py file. Look in __main__ below for more details
def get_orig_uniq_idx_from_col(DBname,COLname):
    client = MongoClient("localhost", 27017)
    raw_idx_id__df = get_raw_idx_id_df(client,DBname,COLname)
    raw_idx_id__df.sort_values('id', inplace=True)
    raw_idx_id__df.reset_index(inplace=True, drop=True)
    uniq_idx = get_uniq_idx(raw_idx_id__df)
    or_idx = get_orig_uniq_idx(client,DBname,COLname)
    uniq_or_idx = list(set(uniq_idx).intersection(set(or_idx)))
    print('Collection contains %d Unique and Original tweets' %len(uniq_or_idx))
    return(uniq_or_idx)
    
    

if __name__ == "__main__":
    # Create a mongoDB client
    client = MongoClient("localhost", 27017)
    # DataBase name, used to store tweets in mongoDB
    DBname = 'Twitter_Assignment'

    # STEP 1
    # Requires LESS THAN 30 secs for each collection
    # Returns a DataFrame with ALL indices and ids of a specific Collection's tweets
    clima_raw_idx_id__df = get_raw_idx_id_df(client,DBname,'Clima_Raw_Tweets')
    
    # STEP 2
    # INSTANT execution
    #Sort DataFrames by id column
    clima_raw_idx_id__df.sort_values('id', inplace=True)
    # IMPORTANT to recreate indexing
    clima_raw_idx_id__df.reset_index(inplace=True, drop=True) 
    
    # STEP 3
    # Requires LESS THAN 10 secs for each collection
    # Imports DataFrames (from previous step) and returns LIST of UNIQUE tweets indices
    clima_uniq_idx = get_uniq_idx(clima_raw_idx_id__df)
    
    # STEP 4
    # Requires LESS THAN 30 secs for each collection
    # Returns a LIST of ORIGINAL (no ReTweets) tweets' indices of a specific Collection 
    clima_or_idx = get_orig_uniq_idx(client,DBname,'Clima_Raw_Tweets')

    # STEP 5
    # INSTANT execution
    # Find untersection of lists of uniques and of original tweet's idx
    # Intersection() is a set function so first turn into sets, complete intersection
    # and then turn result back to list
    clima_uniq_or_idx = list(set(clima_uniq_idx).intersection(set(clima_or_idx)))
