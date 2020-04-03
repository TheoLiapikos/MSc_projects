'''
Manipulation of training dataset obtained by Kaggle:
https://www.kaggle.com/c/twitter-sentiment-analysis2/data

Data fields:
ItemID - id of tweet
Sentiment - sentiment tag
SentimentText - text of the tweet

Num of Samples: ~100.000

TAGS:
0: negative  ~43.500
1: positive  ~56.500
'''

#import os
import pandas as pd
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pickle
import gensim
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.utils import simple_preprocess
from nltk.sentiment.util import mark_negation
from gensim.parsing.preprocessing import STOPWORDS


# Files and Directories used in this part
from files_folders_paths import (path_to_kaggle_train_data,
                                 path_to_kaggle_preproc_train_data
                                 )


# Same preprocess procedure as on rest tweets
def tweet_text_preproc(text):
    result = []
    # Remove hashtags, users and urls from text
    text = re.sub('@\w+',' ',text).strip()
    text = re.sub('http\S+',' ',text).strip()
    text = re.sub('https\S+',' ',text).strip()
    text = re.sub('&\w+',' ',text).strip()
    text = re.sub('#\w+',' ',text).strip()
    # Replace repeated characters with only two of them
    text = re.sub(r'(.)\1+', r'\1\1', text)
    # Remove stop-words
    text = ' '.join([token for token in text.split() if token not in STOPWORDS])
    # Mark negations here before remove special characters such as '.
    # Needs tokenized text, so I split and rejoin tokens
    text = ' '.join(mark_negation(text.split()))
    
    # Deal with positive and negative emoticons
    for emo in positive_emoticons:
        if emo in text:
            result.append('pos_emoticon')
    for emo in negative_emoticons:
        if emo in text:
            result.append('neg_emoticon')
    
    # deacc=True removes punctuations
    proc_text = simple_preprocess(text, deacc=True)
    # Make bigrams
    bigram_mod[proc_text]
    # Make trigrams
    trigram_mod[bigram_mod[proc_text]]
    for token in proc_text:
        if len(token) > 2:
            # Both Lemmatize and Stemm words
#            token =  stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))
            # Just Lemmatize words
            token = WordNetLemmatizer().lemmatize(token, pos='v')
            # Remove stop-words and small words once again after lemmatization
            if(token not in STOPWORDS and len(token) > 2):
                result.append(token)
    return result


# Turn documents into Bag of Words
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))


# Lists with Positive and Negative text emoticons
positive_emoticons = [":D",":)",":-)",":-D",";D",";-)","=D",";)",":]","=)","(:",
                        "XD",":P",";P",":p","=]",":-P","8)","xD","XP",":')"]

negative_emoticons = [":|",":@","D8","DX",":\\","D:",":-/",":/ ",">:(",";(",
                        ":-(",":'(",":("]



if __name__ == "__main__":
    # Dataframe to hold dataset
#    train_df = pd.DataFrame()
    # BEWARE!!! Must be encoding='latin-1', because 'utf-8' encoding raises errors
    train_df = pd.read_csv(path_to_kaggle_train_data, encoding='latin-1')
    # Rename column name to 'text', to match preprocessing function
    train_df.rename(index=str, columns={"SentimentText": "text"}, inplace=True)
    train_df.head(50)
    
    # Drop no needed ItemID column
    train_df.drop('ItemID', inplace=True, axis=1)
    
    # Initial label distribution
    print('Label distribution before preprocessing:')
    print(train_df.Sentiment.value_counts())
    
    
    ##### PREPROCESSING
    data_words = list(sent_to_words(train_df.text))
    
    # Build the bigram and trigram models
    # Higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    
    # The models
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    stemmer = PorterStemmer()
    
    # Preprocess tweets' text and save it to a new column
    train_df['preprocessed_text'] = train_df['text'].map(tweet_text_preproc)
    
    
    # Delete tweets having preprocessed text < 3 words
    # Get indices of tweets to delete
    indices = []
    count = 0
    for text in train_df.preprocessed_text:
        if(len(text)<3):
            indices.append(count)
        count += 1
    
    train_df.drop(train_df.index[indices], inplace=True)
    train_df.reset_index(inplace=True, drop=True)
    print(train_df.head(10))
    
    # Label distribution after preprocessing
    # O: ~34.500   1: ~43.500
    print('\nLabel distribution after preprocessing:')
    print(train_df.Sentiment.value_counts())
    
    # Save preprocessed dataset on disc
    with open(path_to_kaggle_preproc_train_data, "wb") as fp:
        pickle.dump(train_df, fp)
        
    # Delete unnecessary variables
    del data_words, indices, train_df
    


