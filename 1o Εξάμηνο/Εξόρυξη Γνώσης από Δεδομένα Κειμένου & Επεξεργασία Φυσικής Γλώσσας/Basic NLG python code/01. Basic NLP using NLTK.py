# Natural Language Processing using NLTK

# Install NLTK - pip install nltk

import nltk
# Κατεβάζω όλα τα πρόσθετα χρήσιμα πακέτα-data που κάποια θα χρειαστούν στη
# διαδικασία. Απόθηκεύονται στο ~theo και θα πρέπει να σβήσω manual τα .zip 
# αρχεία μετά την αποσυμπίεσή τους. Τελικό μέγεθος ~2.8GB.
# Αλλιώς τα κατεβάζω μεμονωμένα όταν απαιτηθεί
nltk.download('all')

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

# ****************************************************************************
# TOKENIZATION
# ****************************************************************************
# Διαχωρισμός του κειμένου σε παραγράφους ή/και λέξεις. Αντίστοιχα αποτελέσματα
# θα είχαμε και με την εντολή split() της python.
# Tokenizing sentences
sentences = nltk.sent_tokenize(paragraph)
# Tokenizing words
words = nltk.word_tokenize(paragraph)

# ****************************************************************************
# STEMMING
# ****************************************************************************
# Η κλαση που θα κάνει το stemming
from nltk.stem import PorterStemmer

sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer() # Δημιουργία αντικειμένου που θα εκτελέσει το stemming

# Το stemming εφαρμόζεται σε λέξεις, οπότε κάθε πρόταση του κειμένου θα τη διαχωρίσω
# σε λέξεις, θα τις stemmάρω και θα ξαναδημιουργήσω τις προτάσεις
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words]
    sentences[i] = ' '.join(words)

# ****************************************************************************
# LEMMATIZATION
# ****************************************************************************
# Παρόμοια διαδικασία με το stemming
from nltk.stem import WordNetLemmatizer

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words]
    sentences[i] = ' '.join(words)

# ****************************************************************************
# STOP WORDS REMOVAL
# ****************************************************************************
# Αν δεν το έχω ήδη κάνει, κατεβάζω το nltk πακέτο 'stopwords'
# nltk.download('stopwords')
from nltk.corpus import stopwords

sentences = nltk.sent_tokenize(paragraph)

# Removing stopwords. Δηλώνω να χειριστεί μόνο τα english stopwords
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [word for word in words if word not in stopwords.words('english')]
    sentences[i] = ' '.join(words)

# ****************************************************************************
# POS (Part of Speech) TAGGING
# ****************************************************************************
# Ουσιαστικά προσθέτει σε κάθε λέξη του κειμένου μια ταμπέλα με το μέρος του 
# λόγου (ρήμα, επίρρημα, επίθετο κλπ) που είναι

words = nltk.word_tokenize(paragraph)
# Δημιουργείται λίστα με tuples του τύπου (word, POS)
tagged_words = nltk.pos_tag(words)

# Ανασύσταση του αρχικού κειμένου με επισημασμένες τις λέξεις
word_tags = []
for tw in tagged_words:
    word_tags.append(tw[0]+"_"+tw[1])

tagged_paragraph = ' '.join(word_tags)

# ****************************************************************************
# NAMED ENTITY RECOGNITION
# ****************************************************************************
# Ουσιαστικά αναγνωρίζει τις Οντότητες (Χρόνος, Ημερομηνία, Άτομο κλπ) που κρύβονται
# πίσω από κάποιες λέξεις. Οι διαθέσιμες Οντότητες της nltk είναι οι:
"""
ORGANIZATION	    Georgia-Pacific Corp., WHO
PERSON          Eddy Bonte, President Obama
LOCATION	        Murray River, Mount Everest
DATE            June, 2008-06-29
TIME            two fifty a m, 1:30 p.m.
MONEY           175 million Canadian Dollars, GBP 10.40
PERCENT         twenty pct, 18.75 %
FACILITY	        Washington Monument, Stonehenge
GPE	            South East Asia, Midlothian
"""

paragraph2 = "The Taj Mahal was built by Emperor Shah Jahan"
# Εφαρμόζεται σε λέξεις που έχουν υποστεί POS Tagging
words = nltk.word_tokenize(paragraph2)
tagged_words = nltk.pos_tag(words)

# Named Entity Recognition. Ουσιαστικά δημιουργείται μια δομή δέντρου, που για 
# να την ελέγξουμε το σχεδιάζουμε (για απλά-σύντομα κείμενα) 
namedEnt = nltk.ne_chunk(tagged_words)
namedEnt.draw()

# ****************************************************************************
# BAG OF WORDS
# ****************************************************************************
# Ένα Text Representation Model, κατά το οποίο μαζεύονται όλες οι λέξεις όλων
# των εξεταζόμενων documents σε ΕΝΑ σύνολο και υπολογίζεται η συχνότητα εμφάνισης
# της κάθε μίας. Επιλέγεται ένα ποσοστό των πιο συχνά εμφανιζόμενων λέξεων, που
# αποτελούν και το ΤΑΞΙΝΟΜΗΜΕΝΟ λεξικό V. Τέλος, κάθε document αναπαρίσταται με 
# vector μεγέθους V, με τιμές 1 αν περιέχονται στο document οι αντίστοιχες λέξεις
# του V και τιμές 0 διαφορετικά.
import re

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
# Ισοδύναμο των python εντολών:
#freq_words2 = sorted(word2count, key=word2count.get, reverse=True)[:100]
#freq_words3 = sorted(word2count, key=word2count.get, reverse=True)[:100]
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



