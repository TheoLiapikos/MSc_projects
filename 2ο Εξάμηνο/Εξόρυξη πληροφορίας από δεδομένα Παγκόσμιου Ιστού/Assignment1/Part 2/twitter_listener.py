import tweepy
import jsonpickle
import pymongo
from pymongo import MongoClient
import json
import clima_twitter_credentials
import os

# Import paths to Directories and files needed
from files_folders_paths import (path_to_data_dir,
                                 path_to_clima_json
                                 )

# Setup access API
def connect_to_twitter_OAuth(tc):
    auth = tweepy.OAuthHandler(tc.CONSUMER_KEY, tc.CONSUMER_SECRET)
    auth.set_access_token(tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)
    return api


def get_save_tweets_to_file(filepath,api,query,max_tweets,lang='en'):
    tweetCount = 0
    #Open JSON file and save tweets
    with open(filepath, 'w') as f:
        # Send the query
        try:
            for tweet in tweepy.Cursor(api.search,q=query,lang=lang).items(max_tweets):         
                #Convert to JSON format
                f.write(jsonpickle.encode(tweet._json, unpicklable=False) + '\n')
                tweetCount += 1
        except:
            print("Error on tweets' download")
        
        #Display how many tweets we have actually collected
        print("Downloaded %d tweets" %tweetCount)


# This function saves downloaded tweets to MongoDB
def save_tweets_to_mongoDB(client,DBname,COLname,filename):
     # Create a new DB or connect to an existing one
    db = client[DBname]
    # Create a new Collection inside DB or connect to an existing one
    raw_tweets = db[COLname]
    # Read each tweet from file and store it to Collection
    with open(filename, 'r') as f:
        for tweet in f.readlines():
            # Convert to JSON before store to Collection
            raw_tweets.insert_one(json.loads(tweet))


# Delete a Collection from MongoDB
def del_collection(client,DBname,COLname):
    client[DBname][COLname].drop()
    print('Deleted Collection "%s" from DB "%s"' %(COLname, DBname))


# Scan given Collection and return IDXs of unique tweets (no ReTweets, no dublicates)
def check_unique_ids(client,DBname,COLname):
    raw_tweets = client[DBname][COLname]
    ids = []
    for tweet in raw_tweets.find():
        ids.append(tweet['id'])
    ids = set(ids)
    return len(ids)


# return stats data about all Collections in given Database
def check_db_stats(client,DBname):
    # Get all Collection names in DB
    col_names = client[DBname].list_collection_names()
    print('\n"%s" DataBase stats:' %DBname)
    for col in col_names:
        col_entries = client[DBname][col].estimated_document_count()
        col_uniq_ids = check_unique_ids(client,DBname,col)
        print('\nCollection "%s" contains %d entries (%d unique ids)' %(col,col_entries,col_uniq_ids))
        last_tweet = client[DBname][col].find_one(sort=[( '_id', pymongo.DESCENDING )])
        print('Last tweet: "%s"' %last_tweet['text'])



if __name__ == "__main__":
    # Directory to store downloaded tweets
    # If directory doesn't exists create it
    if not os.path.exists(path_to_data_dir):
        try:  
            os.mkdir(path_to_data_dir)
        except OSError:  
            print ('Creation of the directory %s failed.' % path_to_data_dir)
        else:  
            print ('Successfully created the directory %s' % path_to_data_dir)
    else:
        print('Directory %s already exists.' % path_to_data_dir)
    
    # Create a mongoDB client
    client = MongoClient("localhost", 27017)
    # Number of tweets to collect
    max_tweets = 10
    # DataBase name, used to store tweets in mongoDB
    DBname = 'Twitter_Assignment'
    
    # CLIMA HASHTAGS
    if(1):
        clima_query = '#climatechange OR #ClimateChange OR #Climate OR #ClimateEmergency OR' \
        '#GlobalWarming OR #climateemergency OR #ClimateAction'
        # Collection name, used to store tweets in mongoDB
        clima_COLname = 'Clima_Raw_Tweets'
        # Setup access API
        clima_api = connect_to_twitter_OAuth(clima_twitter_credentials)
        get_save_tweets_to_file(path_to_clima_json,clima_api,clima_query,max_tweets)
        save_tweets_to_mongoDB(client,DBname,clima_COLname,path_to_clima_json)
    
    check_db_stats(client,DBname)
    
