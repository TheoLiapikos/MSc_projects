import pandas as pd
from gensim import models
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
np.random.seed(42)
import nltk
nltk.download('wordnet')
import os
import pickle
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors


from files_folders_paths import (path_to_preproc_tweets,
                                 path_to_lda_model_dir,
                                 path_to_lda_data_dictionary,
                                 path_to_lda_topics_wordcloud_dir,
                                 path_to_comb_emerg_topics_list
                                 )


# This method turns 'created_at' field to more usefull format
def modify_dates(tweets_df):
    clear_dates = []
    for tweet in tweets_df.iterrows():
        clear_dates.append(tweet[1].created_at[4:7]+'_'+tweet[1].created_at[8:10])
    return(clear_dates)


# Apply LDA analysis on a list of documents
def lda_analysis(docs, num_of_topics=6):
    # Create the Dictionary used in documents (preprocessed)
    # Returns 20800 (mode 1), 9600 (mode 2) or 25800 (mode 3) words
    dictionary = gensim.corpora.Dictionary(docs)

    # Number of words in Dictionary
    print('Corpus\' Dictionary originally contains: %d words' %len(dictionary))

    # Filtering words in Dictionary
    # Parameters:
    #   no_below (int) – Keep tokens which are contained in at least no_below documents
    #   no_above (float) – Keep tokens which are contained in no more than no_above
    #                      documents (fraction of total corpus size, not an absolute number).
    #   keep_n (int) – Keep only the first keep_n most frequent tokens
    dictionary.filter_extremes(no_below=0, no_above=0.5, keep_n = 500)

    # Number of words in Dictionary after filtering
    print('Dictionary after filtering: %d words' %len(dictionary))

    # For each document we create a dictionary structure reporting how many words
    # and how many times these words appear (ONLY words in Dictionary)
    bow_corpus = [dictionary.doc2bow(doc) for doc in docs]

    ### TF-IDF
    # Create tf-idf model object using gensim.models.TfidfModel on ‘bow_corpus’
    # (tfidf), and then apply transformation to the entire corpus (corpus_tfidf)
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    ####### Running LDA
    lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_of_topics,
                                                 id2word=dictionary,random_state=42,
                                                 passes=20, workers=3)

    # For each topic, explore the words occuring in that topic and its relative weight
    print('\nPrinting topics calculated by algorithm:')
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: %d Word: %s' %(idx, topic))

    return(lda_model, bow_corpus, dictionary)


# Create combined word clouds for a set of related emerging topics
def word_cloud_topics(bin_topics, combo_topic):
        # more colors: 'mcolors.XKCD_COLORS'
#        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  
        stop_colors =  set(STOPWORDS)
        cloud = WordCloud(stopwords=stop_colors,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=20,
                          colormap='tab10',
                          prefer_horizontal=1.0)
        
        fig, axes = plt.subplots(1, len(bin_topics), figsize=(10,10), sharex=True, sharey=True)
        
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(bin_topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title(dates[combo_topic[0][0]+i], fontdict=dict(size=24))
            plt.gca().axis('off')
            
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        return(fig)


# Scan HashTags dictionary structure and return only thw text fields
def clear_hashtags(hashtags_list):
        clr_hashtags = []
        for hashtags in hashtags_list:
            if(len(hashtags) == 0):
                clr_hashtags.append([])
            else:
                hsts = [item['text'] for item in hashtags]
                clr_hashtags.append(hsts)
        return(clr_hashtags)


# Compine tweets' Text and HashTags fields into one field (list)
def text_plus_hashtags(text_list, hashtags_list):
        clear_ht_list = clear_hashtags(hashtags_list)
        combo = []
        for i in range(len(text_list)):
            combo.append(text_list[i] + clear_ht_list[i])
        return(combo)


# Print statistics and detaiked data about detected new Emerging Topics
def report_em_topics(emtpc_combo):
     print('\nNew Emerging Topics statistics:')
     min_d = 100
     max_d = 0
     sum_d = 0
     for emt in emtpc_combo:
         if(len(emt) < min_d):
             min_d = len(emt)
         if(len(emt) > max_d):
             max_d = len(emt)
         sum_d+=len(emt)
     print('\tFound %d new Emerging Topics\n\tmax duration: %d days\n\tmin duration: %d days\n\tmean duration: %.1f days'
          %(len(emtpc_combo), max_d, min_d, sum_d/len(emtpc_combo)))
     
     print('\nDetection of new Emerging Topics (EmTpc) and their evolution in time:')
     for emt in emtpc_combo:
         s_day = dates[emt[0][0]]
         f_day = dates[emt[0][0]+len(emt)-1]
         tpc_idx = emt[0][1]
         print('\tNew EmTpc: %s,  Topic Index: %d,  Duration: %d days,  Termination: %s' 
              %(s_day, tpc_idx, len(emt), f_day))


#########  Choose what data to process: ##########
#   mode=1: Use text field
#   mode=2: Use text & hashtags fields

mode = 1

if(mode == 1): print('\nEmerging Topics Analysis using Text field data\n')
if(mode == 2): print('\nEmerging Topics Analysis using Text & HashTag fields data\n')

if __name__ == "__main__":
    # Directory to hold LDA algorithm's data
    # If store directory doesn't exists create it
    if not os.path.exists(path_to_lda_model_dir):
        try:  
            os.mkdir(path_to_lda_model_dir)
        except OSError:  
            print ('Creation of the directory %s failed.' % path_to_lda_model_dir)
        else:  
            print ('Successfully created the directory %s' % path_to_lda_model_dir)
    else:
        print('Directory %s already exists.' % path_to_lda_model_dir)
    
    # Load extracted preprocessed tweets, only preproc_text and hashtags
    work_df = pd.read_pickle(path_to_preproc_tweets)[['created_at','text_preprocessed',
                            'entities_hashtags']]

    
    # Modify dates' format and replace modified field in dataframe
    work_df['created_at'] = modify_dates(work_df)

    # Extract all dates appearing in 'created_at column'. Reject last value (no
    # needed date)
    dates = sorted(work_df.created_at.unique())[:-1]
    
    # Dictionary structure to hold all data produced by LDA algorithm for each date
    # Data structure:  'date': (model, bow_corpus, dictionary)
    lda_data = {}
    
    # Push dates into dictionary
    lda_data['dates'] = dates

    
    if(mode == 1):
        work_df['text_to_analyze'] = work_df['text_preprocessed']
    if(mode == 2):
        work_df['text_to_analyze'] = text_plus_hashtags(work_df['text_preprocessed'],
                                                       work_df['entities_hashtags'])


    # Each day is a different bin
    for date in dates:
        print('\nOn %s collected %d tweets' %(date, work_df[work_df.created_at == date].shape[0]))
    
        # Text used for LDA analysis    
        processed_docs = work_df[work_df.created_at == date]['text_to_analyze']
        
        #####  LDA analysis of each bin
        lda_data[date] = (lda_analysis(processed_docs, 10))

        
    # Save dictionary with LDA analysis data
    with open(path_to_lda_data_dictionary, "wb") as fp:
        pickle.dump(lda_data, fp)


    # Load LDA data dictionary from disc
#    with open(path_to_lda_data_dictionary, "rb") as fp:
#       lda_data = pickle.load(fp)
    
    
    #####  Emerging Topics analysis
    ## A topic is considered as Emerging Topic only it is correlated to at least
    ## one topic from the next bin
    
    # Compute differences among various LDA models, using Jaccard Distance metric
    # 0: Identical sets, 1: Totally different sets
    threshold = 0.88888    # corresponds to 20% similarity between sets
#    threshold = 0.85714    # corresponds to 25% similarity between sets
#    threshold = 0.82353    # corresponds to 30% similarity between sets
    
    # List to hold emerging topics from all bins
    em_tpcs = []
    
    for i in range(len(dates)-1):
        # Compare i model to i+1
        model_i,_,_ = lda_data[dates[i]]
        model_i_plus1,_,_ = lda_data[dates[i+1]]
        
        mdiff, annotation =  model_i.diff(model_i_plus1, distance='jaccard', num_words=100)
        
        # Check mdiff matrix for distances lower than threshold
        for k in range(len(mdiff)):     # ith model's topics
            for l in range(len(mdiff)): # i+1th model's topics
                if(mdiff[k][l] < threshold):
                    # kth topic of the ith model correlates to the lth topic of
                    # the (i+1)th model
                    em_tpcs.append([(i,k),(i+1,l)])

    ## Join all correlated topics from different bins into one topic 
    combo_em_tpcs = em_tpcs.copy()
    for i in range(len(combo_em_tpcs)-1):
        if(len(combo_em_tpcs[i])==0):
            continue
        to_comp = combo_em_tpcs[i][-1]
        for j in range(i+1, len(combo_em_tpcs)):
            if(len(combo_em_tpcs[j])==0):
                continue
            if(combo_em_tpcs[j][0]==to_comp):
                combo_em_tpcs[i].append(combo_em_tpcs[j][-1])
                to_comp = combo_em_tpcs[j][-1]
                combo_em_tpcs[j] = []
    
    # Clear empty rows. Final emerging topics list
    combo_em_tpcs = [item for item in combo_em_tpcs if len(item)>0]
    
    
    # Store emerging topics list on disc
    with open(path_to_comb_emerg_topics_list, "wb") as fp:
        pickle.dump(combo_em_tpcs, fp)
    
        # Load emerging topics list from disc
#    with open(path_to_comb_emerg_topics_list, "rb") as fp:
#       combo_em_tpcs = pickle.load(fp)
    
    
    ##### Report all detected Emerging Topics, statistics and their duration
    report_em_topics(combo_em_tpcs)
    

    ##### Represent each combo emerging topic with WordClouds
    ## One can execute this last part separately by directly loading lda_data and
    ## combo_em_tpcs files from disc
    
    # Directory to hold word clouds images
    # If store directory doesn't exists create it
    if(0):
        if not os.path.exists(path_to_lda_topics_wordcloud_dir):
            try:  
                os.mkdir(path_to_lda_topics_wordcloud_dir)
            except OSError:  
                print ('Creation of the directory %s failed.' % path_to_lda_topics_wordcloud_dir)
            else:  
                print ('Successfully created the directory %s' % path_to_lda_topics_wordcloud_dir)
        else:
            print('Directory %s already exists.' % path_to_lda_topics_wordcloud_dir)
        
        count = 0
        for em_topic in combo_em_tpcs:
            # For each tuple in list, the first number is the index of date and the second
            # number is the index of the topic in this specific date
            topics = []
            for tup in em_topic:
                d_idx, tpc_idx = tup
                date = dates[d_idx]
                # Get LDA model for specific date
                model,_,_ = lda_data[date]
                # Get required topic
                topic = model.show_topics(formatted=False)[tpc_idx]
                topics.append(topic)
                
            # Get word clouds' figure and save it on disc
            figure = word_cloud_topics(topics, em_topic)
            
            if(mode == 1):
                file_name = os.path.join(path_to_lda_topics_wordcloud_dir, ('text_topic_'+str(count)))
            if(mode == 2):
                file_name = os.path.join(path_to_lda_topics_wordcloud_dir, ('text_hashtags_topic_'+str(count)))
            figure.savefig(file_name, bbox_inches="tight")
            count += 1

