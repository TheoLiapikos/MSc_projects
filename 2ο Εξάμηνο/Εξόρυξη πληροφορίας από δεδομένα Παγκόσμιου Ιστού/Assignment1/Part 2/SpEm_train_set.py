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

# Import functionality to use exactly the same preprocess procedure
#from tweet_preprocessing import tweet_text_preproc

# Files and Directories used in this part
from files_folders_paths import (path_to_spem_train_data_dir,
                                 path_to_spem_mixed_train_data,
                                 path_to_spem_anger_train_data,
                                 path_to_spem_disgust_train_data,
                                 path_to_spem_fear_train_data,
                                 path_to_spem_joy_train_data,
                                 path_to_spem_sadness_train_data,
                                 path_to_spem_surprise_train_data,
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
                                 path_to_spem_preproc_rebal_surprise_train_data
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


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))


# Lists with Positive and Negative text emoticons
positive_emoticons = [":D",":)",":-)",":-D",";D",";-)","=D",";)",":]","=)","(:",
                        "XD",":P",";P",":p","=]",":-P","8)","xD","XP",":')"]

negative_emoticons = [":|",":@","D8","DX",":\\","D:",":-/",":/ ",">:(",";(",
                        ":-(",":'(",":("]


#########  Choose what to do with train datasets: ##########
#   mode=1: Separate emotions train sets from various files
#   mode=2: Preprocess each train dataset text
#   mode=3: Rebalance preprocessed train datasets

mode = 3



####  Separate emotions train sets from various files
if(mode == 1):
    # Dealing with mixed emotions train file
    file = pd.read_csv(path_to_spem_mixed_train_data, sep='\t')

    columns = ['Tweet','anger','disgust','fear','joy','sadness','surprise']
    
    file = file[columns]
    
    #####  Separate Emotions train sets and save to disc
    
    # ANGER
    # Read extra train tweets about ANGER
    anger = pd.read_csv(path_to_spem_train_data_dir+'/2018-anger-regression-train .txt', sep='\t')[['Tweet',]]
    anger['anger'] = int(1)
    anger = anger.append(file[['Tweet','anger']])
    pickle.dump(anger, open(path_to_spem_anger_train_data, "wb"))
    
    # DISGUST
    disgust = file[['Tweet','disgust']]
    pickle.dump(disgust, open(path_to_spem_disgust_train_data, "wb"))
        
    # FEAR
    # Read extra train tweets about FEAR
    fear = pd.read_csv(path_to_spem_train_data_dir+'/2018-fear-regression-train.txt', sep='\t')[['Tweet',]]
    fear['fear'] = int(1)
    fear = fear.append(file[['Tweet','fear']])
    pickle.dump(fear, open(path_to_spem_fear_train_data, "wb"))
    
    # JOY
    # Read extra train tweets about JOY
    joy = pd.read_csv(path_to_spem_train_data_dir+'/2018-joy-regression-train.txt', sep='\t')[['Tweet',]]
    joy['joy'] = int(1)
    joy = joy.append(file[['Tweet','joy']])
    pickle.dump(joy, open(path_to_spem_joy_train_data, "wb"))
    
    # SADNESS
    # Read extra train tweets about SADNESS
    sadness = pd.read_csv(path_to_spem_train_data_dir+'/2018-sadness-regression-train.txt', sep='\t')[['Tweet',]]
    sadness['sadness'] = int(1)
    sadness = sadness.append(file[['Tweet','sadness']])
    pickle.dump(sadness, open(path_to_spem_sadness_train_data, "wb"))
    
    # SURPRISE
    surprise = file[['Tweet','surprise']]
    pickle.dump(surprise, open(path_to_spem_surprise_train_data, "wb"))

    del anger, disgust, fear, joy, sadness, surprise, file



#####  Preprocess emotions tweets' text
if(mode == 2):
## Since I have a lot less tweets to train I will keep all non empty tweets

    emotion_files = [
                    (path_to_spem_anger_train_data, path_to_spem_preproc_anger_train_data),
                    (path_to_spem_disgust_train_data, path_to_spem_preproc_disgust_train_data),
                    (path_to_spem_fear_train_data, path_to_spem_preproc_fear_train_data),
                    (path_to_spem_joy_train_data, path_to_spem_preproc_joy_train_data),
                    (path_to_spem_sadness_train_data, path_to_spem_preproc_sadness_train_data),
                    (path_to_spem_surprise_train_data, path_to_spem_preproc_surprise_train_data)
                    ]
    
    # File to read and file to save results
    for (load_file, save_file) in emotion_files:
        
        # Read dataset from disc
        dataset = pickle.load(open(load_file, "rb"))
        # Dataset's emotion
        emotion = dataset.columns[1]
        
        print('\n\nStart preprocessing dataset for emotion: \'%s\'' %emotion)
        
        # Tokenize each tweet
        data_words = list(sent_to_words(dataset.Tweet))
        
        # Build the bigram and trigram models
        # Higher threshold fewer phrases.
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
        
        # The models
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        trigram_mod = gensim.models.phrases.Phraser(trigram)
        
        stemmer = PorterStemmer()
                
        dataset['preprocessed_text'] = dataset.Tweet.map(tweet_text_preproc)
        
        
        # Delete empty tweets. Get indices of tweets to delete
        indices = []
        count = 0
        for text in dataset.preprocessed_text:
            if(len(text) == 0):
                indices.append(count)
            count += 1
        
        dataset.drop(dataset.index[indices], inplace=True)
        dataset.reset_index(inplace=True, drop=True)
        print(dataset.head(10))
    
        # Label distribution after preprocessing
        print('\nLabel distribution for emotion \'%s\' after preprocessing:' %emotion)
        print(dataset[emotion].value_counts())
        
        # Save preprocessed dataset on disc
        pickle.dump(dataset[['Tweet', 'preprocessed_text', emotion]], open(save_file, "wb"))


    del data_words, dataset



#####  Rebalance preprocessed train datasets
if(mode == 3):
    train_files = [
                    (path_to_spem_preproc_anger_train_data, path_to_spem_preproc_rebal_anger_train_data),
                    (path_to_spem_preproc_disgust_train_data, path_to_spem_preproc_rebal_disgust_train_data),
                    (path_to_spem_preproc_fear_train_data, path_to_spem_preproc_rebal_fear_train_data),
                    (path_to_spem_preproc_joy_train_data, path_to_spem_preproc_rebal_joy_train_data),
                    (path_to_spem_preproc_sadness_train_data, path_to_spem_preproc_rebal_sadness_train_data),
                    (path_to_spem_preproc_surprise_train_data, path_to_spem_preproc_rebal_surprise_train_data)
                    ]

    # File to read and file to save results
    for (load_file, save_file) in train_files:

        # Read dataset from disc
        dataset = pickle.load(open(load_file, "rb"))
        # Dataset's emotion
        emotion = dataset.columns[2]
        print('\n\nRebalancing train dataset for emotion \'%s\':' %emotion)
        # Separate set according to label (0/1)
        X0 = dataset[dataset[emotion] == 0]
        pop0 = len(X0)
        print('\tNumber of samples having label 0: %d' %pop0)
        X1 = dataset[dataset[emotion] == 1]
        pop1 = len(X1)
        print('\tNumber of samples having label 1: %d' %pop1)
        
        print('\nNumber of label 1 samples will increase from %d to %d using resample with replacement:' %(pop1, pop0))
        
        # Reset the index on set with label 1. I will use old indexing to control resampling
        X1 = X1.reset_index()
        X1 = X1.sample(n=pop0, replace=True, random_state=42)
        
        print('\tSize of resampled label 1 set: %d' %len(X1.index))
        print('\tNumber of unique tweets in resampled label 1 set: %d' %X1.shape[0])
        
        # Reunite samples of any label
        X0 = X0.append(X1.drop('index', axis=1))
        print('\nAfter rebalancing training set for emotion \'%s\' contains %d samples' %(emotion,X0.shape[0]))
        
        # Save preprocessed rebalanced dataset on disc
        pickle.dump(X0, open(save_file, "wb"))




