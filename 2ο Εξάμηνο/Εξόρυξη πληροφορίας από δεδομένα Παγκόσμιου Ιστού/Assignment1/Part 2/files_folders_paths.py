import os


###########################  Directories  ###############################
# Data directory
data_dir_name = 'tweets_data'
path_to_data_dir = os.path.join(os.getcwd(), data_dir_name)


# LDA model directory
lda_model_dir_name = 'emerging_topics_LDA'
path_to_lda_model_dir = os.path.join(path_to_data_dir, lda_model_dir_name)
# WordClouds from LDA model topics
lda_topics_wordcloud_dir_name = 'LDA_topics_wordclouds'
path_to_lda_topics_wordcloud_dir = os.path.join(path_to_lda_model_dir, lda_topics_wordcloud_dir_name)


# Overall Sentiment Analysis directories
overall_sent_dir_name = 'Overall_Sentiment_Analysis'
path_to_overall_sent_dir = os.path.join(path_to_data_dir, overall_sent_dir_name)
# Overall Sentiment Analysis Kaggle train data directory
kaggle_train_data_dir_name = 'Kaggle_train_data'
path_to_kaggle_train_data_dir = os.path.join(path_to_overall_sent_dir, kaggle_train_data_dir_name)
# Overall Sentiment Analysis trained_classifiers directory
os_trained_classifiers_data_dir_name = 'trained_classifiers'
path_to_os_trained_classifiers_dir = os.path.join(path_to_overall_sent_dir,
                                                  os_trained_classifiers_data_dir_name)


# Specific Emotions Directories
specific_emotions_dir_name = 'Specific_Emotions_Analysis'
path_to_specific_emotions_dir = os.path.join(path_to_data_dir, specific_emotions_dir_name)
# Specific Emotions train data irectory
spem_train_data_dir_name = 'SpEm_train_data'
path_to_spem_train_data_dir = os.path.join(path_to_specific_emotions_dir, spem_train_data_dir_name)
# pecific Emotions Analysis trained_classifiers directory
spem_trained_classifiers_dir_name = 'spem_trained_classifiers'
path_to_spem_trained_classifiers_dir = os.path.join(path_to_specific_emotions_dir,
                                                  spem_trained_classifiers_dir_name)


# Statistical Analysis Directory
stat_analysis_dir_name = 'Statistical_Analysis_Graphs'
path_to_stat_analysis_dir = os.path.join(path_to_data_dir, stat_analysis_dir_name)


# Geolocation Directory
geoloc_dir_name = 'Geolocation_Analysis' 
path_to_geoloc_dir = os.path.join(path_to_data_dir, geoloc_dir_name)





####################### Various files used  ##############################
clima_json_filename = 'clima_tweets.json'
path_to_clima_json = os.path.join(path_to_data_dir, clima_json_filename)

crypto_json_filename = 'crypto_tweets.json'
path_to_crypto_json = os.path.join(path_to_data_dir, crypto_json_filename)

raw_tweets_name = 'raw_tweets.pkl'
path_to_raw_tweets = os.path.join(path_to_data_dir, raw_tweets_name)

#preproc1_tweets_name = 'preproc1_tweets.pkl'
#path_to_preproc1_tweets = path_to_data_dir + '/' + preproc1_tweets_name

feat_extract1_tweets_name = 'feat_extract1_tweets.pkl'
path_to_feat_extract1_tweets = os.path.join(path_to_data_dir, feat_extract1_tweets_name)

# Files to store preprocessed tweets
preproc_tweets_name = 'preproc_tweets.pkl'
path_to_preproc_tweets = os.path.join(path_to_data_dir, preproc_tweets_name)




# LDA analysis
# File to store LDA models' data (along with bow_corpus and dictionary)
lda_data_dictionary_name = 'LDA_data_dictionary'
path_to_lda_data_dictionary = os.path.join(path_to_lda_model_dir, lda_data_dictionary_name)
# File with compined emerging topics list
comb_emerg_topics_list_name = 'combined_emerging_topics_list'
path_to_comb_emerg_topics_list = os.path.join(path_to_lda_model_dir, comb_emerg_topics_list_name)



#lda_text_model_name = 'text_LDA_model'
#path_to_lda_text_model = os.path.join(path_to_lda_model_dir, lda_text_model_name)
#lda_hashtags_model_name = 'hashtags_LDA_model'
#path_to_lda_hashtags_model = os.path.join(path_to_lda_model_dir, lda_hashtags_model_name)
#lda_text_hash_model_name = 'text_hash_LDA_model'
#path_to_lda_text_hash_model = os.path.join(path_to_lda_model_dir, lda_text_hash_model_name)
#
#lda_text_dictionary_name = 'text_dictionary'
#path_to_lda_text_dictionary = os.path.join(path_to_lda_model_dir, lda_text_dictionary_name)
#lda_hashtags_dictionary_name = 'hashtags_dictionary'
#path_to_lda_hashtags_dictionary = os.path.join(path_to_lda_model_dir, lda_hashtags_dictionary_name)
#lda_text_hash_dictionary_name = 'text_hash_dictionary'
#path_to_lda_text_hash_dictionary = os.path.join(path_to_lda_model_dir, lda_text_hash_dictionary_name)
#
#lda_text_bow_corpus_name = 'text_bow_corpus'
#path_to_lda_text_bow_corpus = os.path.join(path_to_lda_model_dir, lda_text_bow_corpus_name)
#lda_hashtags_bow_corpus_name = 'hashtags_bow_corpus'
#path_to_lda_hashtags_bow_corpus = os.path.join(path_to_lda_model_dir, lda_hashtags_bow_corpus_name)
#lda_text_hash_bow_corpus_name = 'text_hash_bow_corpus'
#path_to_lda_text_hash_bow_corpus = os.path.join(path_to_lda_model_dir, lda_text_hash_bow_corpus_name)





# Files used for Overall Sentiment Analysis
# Kaggle original train data
kaggle_train_data_name = 'train.csv'
path_to_kaggle_train_data = os.path.join(path_to_kaggle_train_data_dir, kaggle_train_data_name)
# Kaggle preprocessed train data
kaggle_preproc_train_data_name = 'kaggle_preproc_train.pkl'
path_to_kaggle_preproc_train_data = os.path.join(path_to_kaggle_train_data_dir, kaggle_preproc_train_data_name)
# Vectorizer trained with train data
os_trained_vectorizer_name = 'os_trained_vectorizer'
path_to_os_trained_vectorizer = os.path.join(path_to_overall_sent_dir, os_trained_vectorizer_name)
# Classification accuracy results dataframe
os_trained_clfs_accuracies_df_name = 'os_trained_clfs_accuracies_df'
path_to_os_trained_clfs_accuracies_df =  os.path.join(path_to_overall_sent_dir, os_trained_clfs_accuracies_df_name)
# Word features used for classifications training
os_training_word_features_name = 'os_training_word_features'
path_to_os_training_word_features =  os.path.join(path_to_overall_sent_dir, os_training_word_features_name)
# DataFrame with tweets' feature representation
os_tweets_feature_representation_name = 'os_tweets_feature_representation'
path_to_os_tweets_feature_representation =  os.path.join(path_to_overall_sent_dir, os_tweets_feature_representation_name)
# DataFrame with tweets' predictions
os_tweets_predictions_name = 'os_tweets_predictions'
path_to_os_tweets_predictions =  os.path.join(path_to_overall_sent_dir, os_tweets_predictions_name)

# Files used for Specific Emotions Analysis
# Mixed train data file
spem_mixed_train_data_name = '2018-11 emotions-classification-train.txt'
path_to_spem_mixed_train_data =  os.path.join(path_to_spem_train_data_dir, spem_mixed_train_data_name)
# Separated emotions train data
spem_anger_train_data_name = 'spem_anger_train_data'
path_to_spem_anger_train_data = os.path.join(path_to_spem_train_data_dir, spem_anger_train_data_name)
spem_disgust_train_data_name = 'spem_disgust_train_data'
path_to_spem_disgust_train_data = os.path.join(path_to_spem_train_data_dir, spem_disgust_train_data_name)
spem_fear_train_data_name = 'spem_fear_train_data'
path_to_spem_fear_train_data = os.path.join(path_to_spem_train_data_dir, spem_fear_train_data_name)
spem_joy_train_data_name = 'spem_joy_train_data'
path_to_spem_joy_train_data = os.path.join(path_to_spem_train_data_dir, spem_joy_train_data_name)
spem_sadness_train_data_name = 'spem_sadness_train_data'
path_to_spem_sadness_train_data = os.path.join(path_to_spem_train_data_dir, spem_sadness_train_data_name)
spem_surprise_train_data_name = 'spem_surprise_train_data'
path_to_spem_surprise_train_data = os.path.join(path_to_spem_train_data_dir, spem_surprise_train_data_name)
# Separated emotions preprocessed train data
spem_preproc_anger_train_data_name = 'spem_preproc_anger_train_data'
path_to_spem_preproc_anger_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_anger_train_data_name)
spem_preproc_disgust_train_data_name = 'spem_preproc_disgust_train_data'
path_to_spem_preproc_disgust_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_disgust_train_data_name)
spem_preproc_fear_train_data_name = 'spem_preproc_fear_train_data'
path_to_spem_preproc_fear_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_fear_train_data_name)
spem_preproc_joy_train_data_name = 'spem_preproc_joy_train_data'
path_to_spem_preproc_joy_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_joy_train_data_name)
spem_preproc_sadness_train_data_name = 'spem_preproc_sadness_train_data'
path_to_spem_preproc_sadness_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_sadness_train_data_name)
spem_preproc_surprise_train_data_name = 'spem_preproc_surprise_train_data'
path_to_spem_preproc_surprise_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_surprise_train_data_name)
# Separated emotions preprocessed rebalanced train data
spem_preproc_rebal_anger_train_data_name = 'spem_preproc_rebal_anger_train_data'
path_to_spem_preproc_rebal_anger_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_anger_train_data_name)
spem_preproc_rebal_disgust_train_data_name = 'spem_preproc_rebal_disgust_train_data'
path_to_spem_preproc_rebal_disgust_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_disgust_train_data_name)
spem_preproc_rebal_fear_train_data_name = 'spem_preproc_rebal_fear_train_data'
path_to_spem_preproc_rebal_fear_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_fear_train_data_name)
spem_preproc_rebal_joy_train_data_name = 'spem_preproc_rebal_joy_train_data'
path_to_spem_preproc_rebal_joy_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_joy_train_data_name)
spem_preproc_rebal_sadness_train_data_name = 'spem_preproc_rebal_sadness_train_data'
path_to_spem_preproc_rebal_sadness_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_sadness_train_data_name)
spem_preproc_rebal_surprise_train_data_name = 'spem_preproc_rebal_surprise_train_data'
path_to_spem_preproc_rebal_surprise_train_data = os.path.join(path_to_spem_train_data_dir, spem_preproc_rebal_surprise_train_data_name)
# Classification accuracy results dataframe for preproc datasets
spem_prepoc_trained_clfs_accuracies_name = 'spem_prepoc_trained_clfs_accuracies_df'
path_to_spem_prepoc_trained_clfs_accuracies_df =  os.path.join(path_to_specific_emotions_dir, spem_prepoc_trained_clfs_accuracies_name)
# Classification accuracy results dataframe for preproc rebalanced datasets
spem_prepoc_rebal_trained_clfs_accuracies_name = 'spem_prepoc_rebal_trained_clfs_accuracies_df'
path_to_spem_prepoc_rebal_trained_clfs_accuracies_df = os.path.join(path_to_specific_emotions_dir, spem_prepoc_rebal_trained_clfs_accuracies_name)
# Files with tweets' predictions
spem_preproc_tweets_predictions_name = 'spem_preproc_tweets_predictions'
path_to_spem_preproc_tweets_predictions = os.path.join(path_to_specific_emotions_dir, spem_preproc_tweets_predictions_name)
spem_preproc_rebal_tweets_predictions_name = 'spem_preproc_rebal_tweets_predictions'
path_to_spem_preproc_rebal_tweets_predictions = os.path.join(path_to_specific_emotions_dir, spem_preproc_rebal_tweets_predictions_name)

# Files used for Geolocation Analysis
# Tweets distribution on World map graph
geoloc_graph_name = 'tweets_geolocation.png'
path_to_geoloc_graph = os.path.join(path_to_geoloc_dir, geoloc_graph_name)
# Coordinates found by Geocoder
geocoder_results_name = 'geocoder_results'
path_to_geocoder_results = os.path.join(path_to_geoloc_dir, geocoder_results_name)
