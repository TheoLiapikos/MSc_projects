# ****************************************************************************
# TF-IDF Model
# ****************************************************************************
# Ένα Text Representation Model, που προσπαθεί να ξεπεράσει τις αδυναμίες του BOW.
# Εφαρμόζεται πάνω σε όλο το λεξιλόγιο V του corpus.
# TF: Term Frequency, # of occur. of a specific word in a specific doc / # of
#     words in the specific doc. Άρα κάθε λέξη έχει από μία τιμή TF για κάθε doc.
# IDF: Inverse Document Frequency, log(# of corpus' docs / # of docs containing 
#     a specific word)+1. Άρα κάθε λέξη έχει μία τιμή IDF για όλο το corpus.
# TF-IDF model: TF * IDF.
# Οι πιο σπάνιες λέξεις θα έχουν μεγαλύτερη τιμή IDF και πριμοδοτούνται σε σχέση
# με τις πιο συχνές λέξεις που εμφανίζονται σε περισσότερα docs.

import nltk
import re
import heapq
import numpy as np

paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year. The Revenant was 
               the product of the tireless efforts of an unbelievable cast
               and crew. First off, to my brother in this endeavor, Mr. Tom 
               Hardy. Tom, your talent on screen can only be surpassed by 
               your friendship off screen … thank you for creating a t
               ranscendent cinematic experience. Thank you to everybody at 
               Fox and New Regency … my entire team. I have to thank 
               everyone from the very onset of my career … To my parents; 
               none of this would be possible without you. And to my 
               friends, I love you dearly; you know who you are. And lastly,
               I just want to say this: Making The Revenant was about
               man's relationship to the natural world. A world that we
               collectively felt in 2015 as the hottest year in recorded
               history. Our production needed to move to the southern
               tip of this planet just to be able to find snow. Climate
               change is real, it is happening right now. It is the most
               urgent threat facing our entire species, and we need to work
               collectively together and stop procrastinating. We need to
               support leaders around the world who do not speak for the 
               big polluters, but who speak for all of humanity, for the
               indigenous people of the world, for the billions and 
               billions of underprivileged people out there who would be
               most affected by this. For our children’s children, and 
               for those people out there whose voices have been drowned
               out by the politics of greed. I thank you all for this 
               amazing award tonight. Let us not take this planet for 
               granted. I do not take tonight for granted. Thank you so very much."""

# Τα αρχικά στάδια μέχρι και το φιλτράρισμα των πιο συχνών λέξεων για τη δημιουργία
# του λεξιλογίου V είναι κοινά με το BOW model

# Διαχωρίζω τις προτάσεις του κειμένου και κάνω preprocessing αφαιρώντας τα μη
# αλφαριθμητικά, τα πολλαπλά κενά και μετατρέποντας τους χαρακτήρες σε πεζά
dataset = nltk.sent_tokenize(paragraph)
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()
    dataset[i] = re.sub(r'\W',' ',dataset[i])
    dataset[i] = re.sub(r'\s+',' ',dataset[i])

# Δημιουργία Ιστογράμματος - Συχνότητα εμφάνισης λέξεων
# Διαχωρίζω τις λέξεις του κειμένου και δημιουργώ δομή λεξικού "word":frequency
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

# Φιλτράρισμα λέξεων: Επιλογή των 100 πιο συχνών λέξεων του κειμένου. Χρησιμοποιώ
# τα εργαλεία της βιβλιοθήκης heapq. Ως key της ταξινόμησης ορίω τη συνάρτηση 
# get των dictionary που επιστρέφει τα values
freq_words = heapq.nlargest(100,word2count,key=word2count.get)
# Ισοδύναμο της python εντολής:
#freq_words2 = sorted(word2count, key=word2count.get, reverse=True)[:100]
# Αν θέλω να διατηρήσω τις συχνότητες των λέξεων η παρακάτω εντολή επιστρέφει
# λίστα με tuples (word, freq):
#freq_dict = sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:100]

# Υλοποίηση του IDF πίνακα, με τη δομή Λεξικού
word_idfs = {}
for word in freq_words:
    doc_count = 0
    for data in dataset:
        if word in nltk.word_tokenize(data):
            doc_count += 1
    word_idfs[word] = np.log(len(dataset)/(1+doc_count))

# Υλοποίηση του TF πίνακα, με τη δομή Λεξικού
tf_matrix = {}
for word in freq_words:
    doc_tf = []
    for data in dataset:
        frequency = 0
        for w in nltk.word_tokenize(data):
            if word == w:
                frequency += 1
        tf_word = frequency/len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf

# Υπολογισμός του TF-IDF Model
tfidf_matrix = []
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)
    tfidf_matrix.append(tfidf)   

# Τελικές ρυθμίσεις του πίνακα του TF-IDF Model
X = np.asarray(tfidf_matrix)

X = np.transpose(X)


