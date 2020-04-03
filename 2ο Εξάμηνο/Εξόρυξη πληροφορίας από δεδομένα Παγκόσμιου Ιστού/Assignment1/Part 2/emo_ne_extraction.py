import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import time
from nltk import ne_chunk, pos_tag, word_tokenize


# Files and Directories used in this part
from files_folders_paths import (path_to_feat_extract1_tweets,
                                 path_to_raw_tweets
                                 )

# Lists with Positive and Negative text emoticons
positive_emoticons = [":D",":)",":-)",":-D",";D",";-)","=D",";)",":]","=)","(:",
                        "XD",":P",";P",":p","=]",":-P","8)","xD","XP",":')"]

negative_emoticons = [":|",":@","D8","DX",":\\","D:",":-/",":/ ",">:(",";(",
                        ":-(",":'(",":("]


# Find named entities using NLTK
# Check if you want to keep just entity's name or its tag too
# Takes about 8 min for about 43.000 tweets
def tweet_text_named_entities(tweet):
    continuous_chunk = []
    for chunk in ne_chunk(pos_tag(word_tokenize(tweet.text))):
        if hasattr(chunk, 'label'):
            entity_name = ' '.join(c[0] for c in chunk.leaves())
            # If I want just entity name
            continuous_chunk.append(entity_name)
            # If I want both entity name and its tag
#            entity_type = chunk.label()
#            continuous_chunk.append((entity_name,entity_type))
    text_named_entities.append(continuous_chunk)


def emos_extract(tweet):
    text = tweet.text
    pos = 0
    neg = 0
    # Check for positive emoticons
    for item in positive_emoticons:
        if item in text:
            pos += 1
    # Check for negative emoticons
    for item in negative_emoticons:
        if item in text:
            neg += 1
    pos_emoticons.append(pos)
    neg_emoticons.append(neg)


# Feature extract tweets
if __name__ == "__main__":

    t0 = time.time()
    # Check if a previous preprocessed1 file exists
    try:
        # Read tweets
        tweets = pd.read_pickle(path_to_raw_tweets)
        
        # New data (columns) to be added from preprocessing
        pos_emoticons = []
        neg_emoticons = []
        text_named_entities = []
        
        count = 0
        print('\nStart extracting features from %d tweets...' %len(tweets))
        for _, tweet in tweets.iterrows():
            count += 1
            emos_extract(tweet)
            tweet_text_named_entities(tweet)
            if(count % 5000 == 0):
                print('5000 tweets interval: %d sec' %(time.time()-t0))
        
        # Add new columns to dataframe
        tweets['pos_emoticons'] = pos_emoticons
        tweets['neg_emoticons'] = neg_emoticons
        tweets['text_named_entities'] = text_named_entities
        
        t1 = time.time()
        mins = (t1-t0) // 60
        secs = (t1-t0) % 60
        print('Fearures extracted from %d tweets in %d:%02d' %(len(tweets), mins, secs))

        # Save Feature extracted1 tweets on disc
        tweets.to_pickle(path_to_feat_extract1_tweets)
        
    except: 
        print('No tweets\' file to extract features')

