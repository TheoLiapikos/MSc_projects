# ****************************************************************************
# BAG OF WORDS
# ****************************************************************************
# Ένα Text Representation Model, κατά το οποίο μαζεύονται όλες οι λέξεις όλων
# των εξεταζόμενων documents σε ΕΝΑ σύνολο και υπολογίζεται η συχνότητα εμφάνισης
# της κάθε μίας. Επιλέγεται ένα ποσοστό των πιο συχνά εμφανιζόμενων λέξεων, που
# αποτελούν και το ΤΑΞΙΝΟΜΗΜΕΝΟ λεξικό V. Τέλος, κάθε document αναπαρίσταται με 
# vector μεγέθους V, με τιμές 1 αν περιέχονται στο document οι αντίστοιχες λέξεις
# του V και τιμές 0 διαφορετικά.
# Μεγάλο μειονέκτημα ότι, αφού τα vectors έχουν τιμές μόνο 0 ή 1, εξισώνεται η
# 'αξία' κάθε λέξης. Ευνοούνται οι πιο συχνές λέξεις, που μπορεί να μην έχουν
# ιδιαίτερη αξία σε σχέση με κάποιες λιγότερο συχνές, αλλά με μεγαλύτερη αξία.
# Την αδυναμία αυτή προσπαθεί να λύσει το TF-IDF model.

import nltk
import re

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
import heapq
freq_words = heapq.nlargest(100,word2count,key=word2count.get)
# Ισοδύναμο της python εντολής:
#freq_words2 = sorted(word2count, key=word2count.get, reverse=True)[:100]
# Αν θέλω να διατηρήσω τις συχνότητες των λέξεων η παρακάτω εντολή επιστρέφει
# λίστα με tuples (word, freq):
#freq_dict = sorted(word2count.items(), key=lambda x: x[1], reverse=True)[:100]

# Στο παράδειγμα το κείμενο είναι χωρισμένο σε 21 sentences, που αντιστοιχούν στα
# documents. Μετατροπή των documents σε vectors μεγέθους V=100.
import numpy as np
X = []
for data in dataset:
    vector = []
    data_words=nltk.word_tokenize(data)
    for word in freq_words:
        if word in data_words:
            vector.append(1)
        else:
            vector.append(0)
    X.append(vector)
# Μετατροπή της λίστα σε 2D array μεγέθους 21x100   
X = np.asarray(X)

