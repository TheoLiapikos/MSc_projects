import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import pickle
import numpy as np
import pandas as pd
import os
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter


from files_folders_paths import (path_to_os_tweets_predictions,
                                path_to_spem_preproc_rebal_tweets_predictions,
                                path_to_feat_extract1_tweets,
                                path_to_stat_analysis_dir
                                )


#########  Choose what to Compute: ##########
#   mode=1: Display real time Overall Sentiment
#   mode=2: Overall Sentiment distribution among various HashTags
#   mode=3: Overall Sentiment distribution among various Named Entities
#   mode=4: WordClouds of Top N Keywords in each Sentiment
#   mode=5: WordClouds of Top N Keywords in each Emotion
#   mode=6: Sentiments' counts per Emotion
#   mode=7: Emotions' counts per Sentiment


mode = 1

if(mode == 1): print('\nDisplaying real time Overall Sentiment\n')
if(mode == 2): print('\nComputing Overall Sentiment distribution among various HashTags\n')
if(mode == 3): print('\nComputing Overall Sentiment distribution among various Named Entities\n')
if(mode == 4): print('\nWordClouds of Top N Keywords in each Sentiment\n')
if(mode == 5): print('\nWordClouds of Top N Keywords in each Emptiom\n')
if(mode == 6): print('\nComputing Sentiments\' counts per Emotion\n')
if(mode == 7): print('\nComputing Emotions\' counts per Sentiment\n')


if __name__ == "__main__":
    # If directory with Statistical Analysis Graph files doesn't exists create it
    if not os.path.exists(path_to_stat_analysis_dir):
        try:  
            os.mkdir(path_to_stat_analysis_dir)
        except OSError:  
            print ('Creation of the directory %s failed.' % path_to_stat_analysis_dir)
        else:  
            print ('Successfully created the directory %s' % path_to_stat_analysis_dir)
    else:
        print('Directory %s already exists.' % path_to_stat_analysis_dir)

    
    ########  Display real time Overall Sentiment
    if(mode == 1):
        style.use("ggplot")
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        
        def tweet_animate(i):
            votes = pickle.load(open(path_to_os_tweets_predictions, "rb"))['vote']
            pos = votes.value_counts()[1]
            neg = votes.value_counts()[0]
            ratio_pos_neg = pos/neg
            print('\n\nPlot Overall Sentiment carried from tweets about \'clima\' hashtags in real time:')
            print('\t%d tweets examined in total' %votes.shape[0])
            print('\t%d tweets assigned to POSITIVE sentiment' %pos)
            print('\t%d tweets assigned to NEGATIVE sentiment' %neg)
            print('\tRatio POS/NEG twees: %0.4f' %ratio_pos_neg)
            
            xar = []
            yar = []
        
            x = 0
            y = 0
        
            for vote in list(votes):
                x += 1
                if vote == 1:
                    y += 1
                elif vote == 0:
                    y -= ratio_pos_neg
        
                xar.append(x)
                yar.append(y)
                
            ax1.clear()
            ax1.plot(xar,yar)
        
        ani = animation.FuncAnimation(fig, tweet_animate, interval=1000)
        plt.show()
        
    
    ########  Sentiments per Hashtag 
    if(mode == 2):
        data = pickle.load(open(path_to_feat_extract1_tweets, "rb"))[['entities_hashtags']]
    
        # Extracting Hashtags
        extr_hashs = []
        for hashes in data.entities_hashtags:
            hts = [item['text'].lower() for item in hashes]
            extr_hashs.append(hts)
        
        # Find most frequent HashTags
        hc = Counter()
        for item in extr_hashs:
            hc.update(item)
        
        # Choose a variety of HashTags, not only the top most frequent
        hashtags = [item[0] for item in hc.most_common(50)[0:50:5]]
        
        # Find indices for each selected hashtag
        ht_idxs = []
        for item in hashtags:
            idxs = []
            for i in range(len(extr_hashs)):
                if(item in extr_hashs[i]):
                    idxs.append(i)
            ht_idxs.append((item,idxs))
        
        # Structure holding POS/NEG results
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['vote']]
        
        # Get POS/NEG for selected hashtags (percentage counts)
        pc_hashes = []
        pc_posCounts = []
        pc_negCounts = []
        for ht, idxs in ht_idxs:
            tweets = df.iloc[idxs]
            pos = tweets.vote.value_counts()[1]
            neg = tweets.vote.value_counts()[0]
            total = pos + neg
            pc_posCounts.append(round(100*pos/total))
            pc_negCounts.append(round(100*neg/total))
            pc_hashes.append(ht)
        
        # Number of categories on axis x
        N = len(pc_hashes)
        
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(ind, pc_negCounts, width, color = 'red')
        p2 = plt.bar(ind, pc_posCounts, width, bottom=pc_negCounts, color = 'blue')
        
        plt.ylabel('Sentiment (percentage)')
        plt.title('Distribution of Sentiment (Pos/Neg) among selected HashTags')
        plt.xticks(ind, pc_hashes, rotation='vertical')
        #plt.yticks(np.arange(0, 14001, 2000))
        plt.legend((p1[0], p2[0]), ('Negative','Positive'))
        
        # Save image to disc
        filename = os.path.join(path_to_stat_analysis_dir, 'Sents_by_hashtags.png')
        plt.savefig(filename, bbox_inches='tight')
    
    
    ########  Sentiments per Named Entity 
    if(mode == 3):
        data = pickle.load(open(path_to_feat_extract1_tweets, "rb"))[['entities_hashtags','text_named_entities']]
    
        # Extracting list of all unique Hashtags
        extr_hashs = []
        for hashes in data.entities_hashtags:
            hts = [item['text'] for item in hashes]
            extr_hashs.extend(hts)
        extr_hashs = list(set(extr_hashs))
        
        # Extracting Named Entities as one list
        extr_nes = []
        for nes in data.text_named_entities:
            extr_nes.extend(nes)
        
        # Remove any appearances of HashTags
        clean_nes = [x for x in extr_nes if x not in extr_hashs]
        
        
        # Find most frequent Named Entities
        hc = Counter(clean_nes)
    
        
        # Choose some random Named Entities
#        nes = [item[0] for item in hc.most_common(55)[4:14]]
        # Choose some specific Named Entities
        nes = ['Bill Shorten', 'Scott Morrison', 'David Attenborough', 'PhilipHammondUK',
               'AmberRuddHR', 'BrandonLewis', 'UK Parliament', 'Green New Deal',
               'BBC News', 'GregClarkMP']
        
        # Extracting Named Entities as list of lists
        extr_nes = []
        for hashes in data.text_named_entities:
            extr_nes.append(hashes)
        
        
        # Find indices for each selected Named Entity
        ne_idxs = []
        for item in nes:
            idxs = []
            for i in range(len(extr_nes)):
                if(item in extr_nes[i]):
                    idxs.append(i)
            ne_idxs.append((item,idxs))
        
        # Structure holding POS/NEG results
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['vote']]
        
        # Get POS/NEG for selected hashtags (percentage counts)
        ne_hashes = []
        ne_posCounts = []
        ne_negCounts = []
        for ht, idxs in ne_idxs:
            tweets = df.iloc[idxs]
            try:
                pos = tweets.vote.value_counts()[1]
            except:
                pos = 0;
            try:
                neg = tweets.vote.value_counts()[0]
            except:
                neg = 0;
            total = pos + neg
            ne_posCounts.append(round(100*pos/total))
            ne_negCounts.append(round(100*neg/total))
            ne_hashes.append(ht)
        
        # Number of categories on axis x
        N = len(ne_hashes)
        
        ind = np.arange(N)    # the x locations for the groups
        width = 0.35       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(ind, ne_negCounts, width, color = 'red')
        p2 = plt.bar(ind, ne_posCounts, width, bottom=ne_negCounts, color = 'blue')
        
        plt.ylabel('Sentiment (percentage)')
        plt.title('Distribution of Sentiment (Pos/Neg) among selected Named Entities')
        plt.xticks(ind, ne_hashes, rotation='vertical')
        #plt.yticks(np.arange(0, 14001, 2000))
        plt.legend((p1[0], p2[0]), ('Negative','Positive'),loc='upper center')
        
        # Save image to disc
        filename = os.path.join(path_to_stat_analysis_dir, 'Sents_by_nes.png')
        plt.savefig(filename, bbox_inches='tight')
    
    
    
    ########  WordClouds of Top N Keywords in each Sentiment
    if(mode == 4):
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['text_preprocessed','vote']]
        df2 = pickle.load(open(path_to_spem_preproc_rebal_tweets_predictions+'_only_Votes', "rb"))
        
        pos_docs = df['text_preprocessed'][df['vote'] == 1]
        neg_docs = df['text_preprocessed'][df['vote'] == 0]
        
        documents = []
        
        # Join all documents about a specific Sentiment
        documents.append(('Positive', ' '.join(pos_docs)))
        documents.append(('Negative', ' '.join(neg_docs)))

        # Max number of words to appear on Word Cloud
        max_words = 75
        wordcloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=max_words,
                          colormap='tab10',
                          prefer_horizontal=1.0)
        
        for name, text in documents:
            wordcloud.generate(text)
            print('\nWordCloud plot for %d most frequent words appearring in tweets assigned to Sentiment: \'%s\'' %(max_words, name))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            # Save Word Cloud image
            filename = os.path.join(path_to_stat_analysis_dir, ('WC_for_Sent_'+name))
            plt.savefig(filename)
            plt.show()
 
        
        del df,df2,pos_docs,neg_docs
    
    
    ########  WordClouds of Top N Keywords in each Emotion
    if(mode == 5):
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['text_preprocessed','vote']]
        df2 = pickle.load(open(path_to_spem_preproc_rebal_tweets_predictions+'_only_Votes', "rb"))

        anger_docs = df2['text_preprocessed'][df2['anger_vote'] == 1]
        disgust_docs = df2['text_preprocessed'][df2['disgust_vote'] == 1]
        fear_docs = df2['text_preprocessed'][df2['fear_vote'] == 1]
        joy_docs = df2['text_preprocessed'][df2['joy_vote'] == 1]
        sadness_docs = df2['text_preprocessed'][df2['sadness_vote'] == 1]
        surprise_docs = df2['text_preprocessed'][df2['surprise_vote'] == 1]
        
        documents = []

        # Join all documents about a specific Emotion
        documents.append(('Anger' ,' '.join(anger_docs)))
        documents.append(('Disgust' ,' '.join(disgust_docs)))
        documents.append(('Fear' ,' '.join(fear_docs)))
        documents.append(('Joy' , ' '.join(joy_docs)))
        documents.append(('Sadness' ,' '.join(sadness_docs)))
        documents.append(('Surprise' ,' '.join(surprise_docs)))
        
        max_words = 75
        wordcloud = WordCloud(background_color='white',
                          width=2500,
                          height=1800,
                          max_words=max_words,
                          colormap='tab10',
                          prefer_horizontal=1.0)
        
        for name, text in documents:
            wordcloud.generate(text)
            print('\nWordCloud plot for %d most frequent words appearring in tweets assigned to Emotion: \'%s\'' %(max_words, name))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            # Save Word Cloud image
            filename = os.path.join(path_to_stat_analysis_dir, ('WC_for_Emotion_'+name))
            plt.savefig(filename)
            plt.show()

        
        del df,df2,anger_docs,disgust_docs,fear_docs,joy_docs,sadness_docs,surprise_docs,documents
    
    
    
    ########  StackedBar Graphs Sentiments counts per Emotion and Emotions counts per Sentiment
    
    ### Sentiments' counts per Emotion
    if(mode == 6):
        # Number of categories on axis x
        N = 6
        posCounts = []
        negCounts = []
        
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['text_preprocessed','vote']]
        df2 = pickle.load(open(path_to_spem_preproc_rebal_tweets_predictions+'_only_Votes', "rb"))
        
        
        anger_idx = df2[df2['anger_vote'] == 1].index
        anger_sentiments = df.iloc[anger_idx]['vote'].value_counts()
        posCounts.append(anger_sentiments[1])
        negCounts.append(anger_sentiments[0])
        
        disgust_idx = df2[df2['disgust_vote'] == 1].index
        disgust_sentiments = df.iloc[disgust_idx]['vote'].value_counts()
        posCounts.append(disgust_sentiments[1])
        negCounts.append(disgust_sentiments[0])
        
        fear_idx = df2[df2['fear_vote'] == 1].index
        fear_sentiments = df.iloc[fear_idx]['vote'].value_counts()
        posCounts.append(fear_sentiments[1])
        negCounts.append(fear_sentiments[0])
        
        joy_idx = df2[df2['joy_vote'] == 1].index
        joy_sentiments = df.iloc[joy_idx]['vote'].value_counts()
        posCounts.append(joy_sentiments[1])
        negCounts.append(joy_sentiments[0])
        
        sadness_idx = df2[df2['sadness_vote'] == 1].index
        sadness_sentiments = df.iloc[sadness_idx]['vote'].value_counts()
        posCounts.append(sadness_sentiments[1])
        negCounts.append(sadness_sentiments[0])
        
        surprise_idx = df2[df2['surprise_vote'] == 1].index
        surprise_sentiments = df.iloc[surprise_idx]['vote'].value_counts()
        posCounts.append(surprise_sentiments[1])
        negCounts.append(surprise_sentiments[0])
        
        ind = np.arange(N)    # the x locations for the groups
        width = 0.55       # the width of the bars: can also be len(x) sequence
        
        p1 = plt.bar(ind, negCounts, width, color = 'red')
        p2 = plt.bar(ind, posCounts, width, bottom=negCounts, color = 'blue')
        
        plt.ylabel('Sentiment counts')
        plt.title('Sentiment (Positive/Negative) counts by Emotion')
        plt.xticks(ind, ('Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'))
        plt.yticks(np.arange(0, 14001, 2000))
        plt.legend((p1[0], p2[0]), ('Negative','Positive'))
        
        # Save image to disc
        filename = os.path.join(path_to_stat_analysis_dir, ('Sent_counts_per_Emotion.png'))
        plt.savefig(filename, bbox_inches='tight')
        
        del df,df2
    
    
    ### Emotions counts per Sentiment
    if(mode == 7):
        emotions = ['anger','disgust','fear','joy','sadness','surprise']
        
        # Structure to save Emotion counts for each Sentiment
        positive_counts = []
        negative_counts = []
        
        df = pickle.load(open(path_to_os_tweets_predictions, "rb"))[['text_preprocessed','vote']]
        df2 = pickle.load(open(path_to_spem_preproc_rebal_tweets_predictions+'_only_Votes', "rb"))
        
        # First analyze POSITIVE tweets
        pos_idx = df[df['vote'] == 1].index
        pos_emotions = df2.iloc[pos_idx]
        
        positive_counts.append(pos_emotions['anger_vote'].value_counts()[1])
        positive_counts.append(pos_emotions['disgust_vote'].value_counts()[1])
        positive_counts.append(pos_emotions['fear_vote'].value_counts()[1])
        positive_counts.append(pos_emotions['joy_vote'].value_counts()[1])
        positive_counts.append(pos_emotions['sadness_vote'].value_counts()[1])
        positive_counts.append(pos_emotions['surprise_vote'].value_counts()[1])
        
        
        # Then NEGATIVE tweets
        neg_idx = df[df['vote'] == 0].index
        neg_emotions = df2.iloc[neg_idx]
        
        negative_counts.append(neg_emotions['anger_vote'].value_counts()[1])
        negative_counts.append(neg_emotions['disgust_vote'].value_counts()[1])
        negative_counts.append(neg_emotions['fear_vote'].value_counts()[1])
        negative_counts.append(neg_emotions['joy_vote'].value_counts()[1])
        negative_counts.append(neg_emotions['sadness_vote'].value_counts()[1])
        negative_counts.append(neg_emotions['surprise_vote'].value_counts()[1])
    
        # All counts into a DataFrame
        dfx = pd.DataFrame(np.asarray([positive_counts, negative_counts]),index=['Positive', 'Negative'], columns=['anger','disgust','fear','joy','sadness','surprise'])
        
        # Plot directly from Datafraem
        dfx.plot.bar(stacked=True, rot=0)
        plt.title('Emotion counts by Sentiment')
        plt.ylabel('Emotion counts')
        # Save image to disc
        filename = os.path.join(path_to_stat_analysis_dir, ('Emotion_counts_per_Sent.png'))
        plt.savefig(filename, bbox_inches='tight')

        del df,df2,pos_emotions,neg_emotions








#votes = pickle.load(open(path_to_os_tweets_predictions, "rb"))['vote']
#xar = []
#yar = []
#
#x = 0
#
#for i in range(len(votes)//100):
#    x += 1
#    part = votes.iloc[i*100:(i+1)*100]
#    pos = part.value_counts()[1]
#    neg = part.value_counts()[0]
#    ratio = pos/neg
##    if vote == 1:
##        y += 1
##    elif vote == 0:
##        y -= ratio_pos_neg
#
#    xar.append(x)
#    yar.append(ratio)
#    
#ax1.clear()
#ax1.plot(xar,yar)
#ax1.show()
#


