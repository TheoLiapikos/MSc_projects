# ****************************************************************************
# Latent Semantic Analysis (LAS)
# ****************************************************************************
# Μοντέλο που δίνει τη δυνατότητα να διαχωριστούν docs σε διάφορες κατηγορίες-
# έννοιες (concepts) με βάση τα χαρακτηριστικά (features) τους.

# Οι χρησιμοποιούμενες βιβλιοθήκες
# Η κλάση TfidfVectorizer εξάγει αυτόματα το TF-IDF model από το corpus
from sklearn.feature_extraction.text import TfidfVectorizer
# Η κλάση TruncatedSVD αποδομεί ένα matrix με το μοντέλο SVD
from sklearn.decomposition import TruncatedSVD
import nltk

# Το corpus αποτελείται από 7 διαφορετικές προτάσεις (docs)
dataset = ["The amount of polution is increasing day by day",
           "The concert was just great",
           "I love to see Gordon Ramsay cook",
           "Google is introducing a new technology",
           "AI Robots are examples of great technology present today",
           "All of us were singing in the concert",
           "We have launch campaigns to stop pollution and global warming"]

# Preprocessing. Το μόνο που απαιτείται σε αυτό το απλό corpus είναι η μετατροπή
# των χαρακτήρων σε πεζά.
dataset = [line.lower() for line in dataset]

# Δημιουργία του TF-IDF model από το corpus. Δημιουργείται 7x42 sparse matrix
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset)

# Visualizing the TF-IDF model. Δίνει τα αποτελέσματα για όλα τα docs. Η πρώτη
# τιμή στο tuple αναφέρεται στο doc (γραμμή), η δεύτερη τιμή αναφέρεται στη θέση
# του κάθε feature (word) στον πίνακα (στήλη) και μετά είναι η TF-IDF τιμή.
# Μπορώ να πάρω τα αποτελέσματα για συγκεκριμένο doc, πχ print(X[3])
print(X)

# Αποδόμηση του TF-IDF matrix με το μοντέλο SVD. Η παράμετρος n_components 
# αναφέρεται στο πλήθος των concepts. Επειδή χρησιμοποιείται εξ' ορισμού ο
# randomized SVD solver, θέτω και τιμή random_state για επαναληψιμότητα
lsa = TruncatedSVD(n_components = 4, n_iter = 100, random_state=42)
lsa.fit(X)

# ΠΡΟΣΟΧΗ. Η μέθοδος μου επιστρέφει μόνο τον V(rxn) αντεστραμμένο πίνακα του SVD.
# Οι r γραμμές αντιστοιχούν στα ζητούμενα concepts και οι n στήλες στα features.
rows = lsa.components_
# Μπορώ να ζητήσω και μεμονωμένο concept: 
# row1 = lsa.components_[0]

# Visualizing the concepts
# Με τον παρακάτω κώδικα πετυχαίνω να επιστρέφονται για κάθε concept ξεχωριστά
# οι 10 πιο σημαντικές λέξεις (που εμφανίζουν την μεγαλύτερη τιμή στον V(rxn))
terms = vectorizer.get_feature_names() # Τα ονόματα των features (λέξεις)
for i,comp in enumerate(lsa.components_): # Για κάθε concept (γραμμή)
    componentTerms = zip(terms,comp) # Δημιουργία tuple (λέξη, τιμή στο concept)
    # Ταξινόμηση φθίνουσα, με βάση την τιμή των λέξεων στο concept
    sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
    sortedTerms = sortedTerms[:10] # Κρατώ μόνο τα 10 πρώτα
    print("\nConcept",i,":")
    for term in sortedTerms:
        print(term)

# Το παραπάνω δεν έχει μεγάλη αξία αφού αυτό που κυρίως μας ενδιαφέρει είναι να
# αντιστοιχίσουμε στα concepts κάθε συνολική πρόταση (sentence)
# Δομή dictionary όπου για κάθε concept θα αποθηκευτούν οι 10 πιο σημαντικές λέξεις
concept_words = {}

# Διαδικασία όπως και παραπάνω
terms = vectorizer.get_feature_names()
for i,comp in enumerate(lsa.components_):
    componentTerms = zip(terms,comp)
    sortedTerms = sorted(componentTerms,key=lambda x:x[1],reverse=True)
    sortedTerms = sortedTerms[:10]
    concept_words["Concept "+str(i)] = sortedTerms

# Υπολογισμός της τιμής που εμφανίζει κάθε πρόταση σε κάθε concept
# Για κάθε concept, παίρνω τις λέξεις κάθε πρότασης, βρίσκω ποιες από αυτές ανήκουν
# στις 10 αημαντικότερες του concept, αθροίζω την τιμή τους και το νούμερο που 
# προκύπτει είναι η τιμή της συγκεκριμένης sentence στο συγκεκριμένο concept
for key in concept_words.keys():
    sentence_scores = []
    for sentence in dataset:
        words = nltk.word_tokenize(sentence)
        score = 0
        for word in words:
            for word_with_score in concept_words[key]:
                if word == word_with_score[0]:
                    score += word_with_score[1]
        sentence_scores.append(score)
    print("\n"+key+":")
    for sentence_score in sentence_scores:
        print(sentence_score)
