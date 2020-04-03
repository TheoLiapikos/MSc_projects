import warnings
warnings.filterwarnings("ignore")
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
import time
import os
from nltk.sentiment.util import mark_negation
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS


# Files and Directories used in this part
from files_folders_paths import (path_to_feat_extract1_tweets,
                                 path_to_preproc_tweets
                                 )


# Less than 1 min for 43.000 tweets
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

# Lists with Positive and Negative text emoticons
positive_emoticons = [":D",":)",":-)",":-D",";D",";-)","=D",";)",":]","=)","(:",
                        "XD",":P",";P",":p","=]",":-P","8)","xD","XP",":')"]

negative_emoticons = [":|",":@","D8","DX",":\\","D:",":-/",":/ ",">:(",";(",
                        ":-(",":'(",":("]

# Preprocess tweets
if __name__ == "__main__":
    # Files and Directories used in this part
    # Data directory
    data_dir_name = 'tweets_data'
    path_to_data_dir = os.getcwd() + '/' + data_dir_name

    t0 = time.time()
    # Tweets to be preprocessed
    to_preproc_tweets = pd.read_pickle(path_to_feat_extract1_tweets)
    
    # Check if a previous preprocessed file already exists
    try:    
        already_preproc_tweets = pd.read_pickle(path_to_preproc_tweets)
    except: # Or else create a new file 
        already_preproc_tweets = pd.DataFrame()
    if(len(to_preproc_tweets)<=len(already_preproc_tweets)):
        print('No new tweets to preprocess!')
    else:
        stemmer = PorterStemmer()
        
        # Portion of total tweets to be preprocessed
        tweets = to_preproc_tweets[len(already_preproc_tweets):]
        
        print('\nStart preprocessing %d new tweets...' %len(tweets))
        
        # Creating Bigram and Trigram Models
        # 1. Tokenize words and Clean-up text
        def sent_to_words(sentences):
            for sentence in sentences:
                yield(simple_preprocess(str(sentence), deacc=True))
        
        data_words = list(sent_to_words(tweets.text))
        
        # 2. Build the bigram and trigram models
        # Higher threshold fewer phrases.
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        
        # The models
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        stemmer = PorterStemmer()
                
        tweets['text_preprocessed'] = tweets['text'].map(tweet_text_preproc)
        
        ### ΙΣΩΣ ΒΑΛΩ ΕΔΩ ΝΑ ΑΦΑΙΡΟΥΝΤΑΙ TWEETS ΜΕ ΚΕΙΜΕΝΟ ΜΙΚΡΟΤΕΡΟ ΤΩΝ 3 ΛΕΞΕΩΝ
        # FROM TRAIN DATA PREPROCESSING
        # Delete tweets having preprocessed text < 3 words
#        # Get indices of tweets to delete
#        indices = []
#        count = 0
#        for text in train_df.preprocessed_text:
#            if(len(text)<3):
#                indices.append(count)
#            count += 1
#        
#        train_df.drop(train_df.index[indices], inplace=True)
#        print(train_df.head(10))
        
        tweets = already_preproc_tweets.append(tweets, ignore_index=1)
        
        # Save preprocessed tweets on disc
        tweets.to_pickle(path_to_preproc_tweets)
        
        t1 = time.time()
        mins = (t1-t0) // 60
        secs = (t1-t0) % 60
        print('Total %d tweets preprocessed in %d:%02d' %(len(tweets), mins, secs))

        # Delete unnecessary variables
        del to_preproc_tweets, data_words



