#!/usr/bin/env python3

#*****************************************************************************
#     Υλοποίηση αλγορίθμων Multinomial Naive Bayes και Binary Naive Bayes
# 
# Η υλοποίηση, κυρίως το κομμάτι των αναγκαίων μαθηματκών υπολογισμών, βασίστηκε
# στα στοιχεία που δίνονται στο paper 'Spam Filtering with Naive Bayes – Which
# Naive Bayes?' και στο βιβλίο 'Jurafsky, Dan & Martin, James H. - Speech and
# Language Processing, 3rd edition'. 
# Μέθοδοι που είναι κοινοί με υπόλοιπους αλγορίθμους, βρίσκονται αποθηκευμένοι
# στο αρχείο 'commonModules.py', απ' όπου και εισάγονται.
#
#*****************************************************************************

import pandas as pd
import numpy as np
from commonModules import preprocDF

# Ορισμός τνω υπερπαραμέτρων του Αλγορίθμου
binNB = False  # Χρήση της Binary εκδοχής
neg = True  # Χειρισμός αρνήσεων


#*****************************************************************************
#        ΤΑ ΔΕΔΟΜΕΝΑ ΕΙΣΟΔΟΥ
#*****************************************************************************
dataset = ['Just plain boring.',
           'Entirely predictable and lacks energy.',
           'No surprises and very few laughs.',
           'Very powerful.',
           'The most fun film of the summer.']

input_data_labels = [0, 0, 0, 1, 1]
test_data = ['predictable with no fun','The most fun film of the summer']

# Δημιουργία DataFrames από τα δεδομένα εισόδου
inputDF = pd.DataFrame({'Sentences':dataset})
inputDF['Labels']  = input_data_labels
testDF = pd.DataFrame({'Sentences':test_data})
# Το σύνολο των διακριτών κλάσεων των δεδομένων
classes = inputDF['Labels'].unique()


#*****************************************************************************
#        PREPROCESSING ΤΩΝ ΔΕΔΟΜΕΝΩΝ ΕΙΣΟΔΟΥ
#*****************************************************************************
# Την απαραίτητη μέθοδο την εισάγω από το αρχείο commonModules.py

preprocDF(inputDF, negations=neg)
preprocDF(testDF, negations=neg)


#*****************************************************************************
#        ΕΚΠΑΙΔΕΥΣΗ ΤΟΥ ΜΟΝΤΕΛΟΥ
#*****************************************************************************
# Υπολογισμός απαραίτητων μεγεθών του μοντέλου

# Η πιθανότητα εμφάνισης (logprior) κάθε κλάσης
logprior = pd.DataFrame()
# Το συνολικό λεξιλόγιο του corpus (μοναδικές λέξεις)
voc = []
# Η loglikelihood πιθανότητα κάθε λέξης να ανήκει σε κάθε κλάση
loglikelihood = pd.DataFrame()


def multinomialNB(inputData, binaryNB=False):
    '''
    Μέθοδος υλοποίησης του αλγορίθμου Multinomial Naive Bayes, τόσο στην κλασική
    όσο και στην Binary εκδοχή του. Τα δεδομένα εισόδου (επισημασμένα κείμενα)
    χρησιμοποιούνται για την εκπαίδευση του μοντέλου.
    \nΠαράμετροι:
     - inputData: Τα δεδομένα εισόδου (κείμενο) σε δομή DataFrame.
     - binaryNB: boolean, optional (default=False). Κατά πόσο θα εφαρμοστεί η 
     Binary έκδοση του αλγορίθμου.
     
    \n\nΕπιστρέφει:
    - Το λεξιλόγιο που παράγεται από τα έγγραφα εκπαίδευσης.
    - Η logprior πιθανότητα εμφάνισης κάθε κλάσης στα γγραφα εκπαδευσης
    - Η loglikelihood πιθανότητα κάθε λέξης να ανήκει σε κάθε κλάση
           
    :Example:
    >>> multinomialNB(dataDF, binaryNB=True)
    >>> 
    '''

    # Υπολογισμός απαραίτητων μεγεθών του μοντέλου
    # Συνολικό πλήθος δεδομένων (προτάσεων) εισόδου
    ndoc = len(inputData)

    # Η πιθανότητα εμφάνισης (logprior) κάθε κλάσης
    logp_c = pd.DataFrame()
    for c in classes:
        # πλήθος προτάσεων της κλάσης
        nc = inputData['Labels'][inputData['Labels'] == c].count()
        # Πιθανότητα εμφάνισης της κλάσης λογαριθμισμένη
        logp_c[c] = [float(np.log(nc/ndoc))] # ΠΡΟΣΟΧΗ Χρησιμοποιείται ΦΥΣΙΚΟΣ λογάριθμος
    logprior = logp_c

    # Το συνολικό λεξιλόγιο του corpus (μοναδικές λέξεις)
    # Πρέπει να έχει προηγηθεί tokenization των προτάσεων σε λέξεις
    v = []
    for doc in inputData['Sentences']:
        v.extend([w for w in doc])
    # Μετατρέπω ξανά σε λίστα για να μπορώ να προσπελαύνω με indexing
    voc = list(set(v))

    # Λεξιλόγιο των κειμένων κάθε κλάσης. ΔΙΑΦΟΡΟΠΟΙΗΣΗ στους 2 αλγορίθμους
    # Multinomial NB:
    # Με ενδιαφέρει το σύνολο των λέξεων (όχι μοναδικές), που εμφανίζεται σε κάθε
    # κείμενο κάθε κλάσης. Απαιτείται η συχνότητα εμφάνισης της κάθε λέξης, οπότε
    # τις μαζεύω όλες σε μια κοινή λίστα (BoW) πριν τις αποθηκεύσω στο DataFrame.
    # Binary Multinomial NB:
    # Με ενδιαφέρουν οι μοναδικές λέξεις που εμφανίζονται σε κάθε ένα doc της
    # κλάσης ξεχωριστά. Έτσι μια λέξη μπορεί να εμφανίζεται πολλές φορές στο
    # σύνολο των λέξεων κάθε κλάσης αλλά το πολύ μία φορά σε κάποιο doc της κλάσης.
    # Τις λέξεις κάθε doc τις κάνω ΣΥΝΟΛΟ (set) πριν τις ενώσω και τις αποθηκεύσω
    # στο DataFrame.
    words_c = pd.DataFrame()
    for c in classes:
        # τα δεδομένα εισόδου της κλάσης
        doc_c = inputData['Sentences'][inputData['Labels'] == c]
        vc = []
        for doc in doc_c:
            if (binaryNB):
                vc.extend(set([w for w in doc]))
            else:
                vc.extend([w for w in doc])
        words_c[c] = [vc]

    # Υπολογισμός της loglikelihood πιθανότητας κάθε λέξης να ανήκει σε κάθε κλάση
    # Αρχική αποθήκευση δεδομένων σε δομή λεξικού
    log_llh = {'V':[]}
    for word in v:
        log_llh['V'].append(word)
    for c in classes:
        log_llh[c]= []
        for word in v:
            # Μετρώ πόσες φορές εμφανίζεται η λέξη στα docs της κλάσης c
            count = 0
            for w in words_c[c]:
                if w == word:
                    count += 1
            # Υπολογισμός της loglikelihood πιθανότητας της λέξης σύμφωνα με τον
            # τύπο του βιβλίου. Χρησιμοποιείται add-1 (Laplace) smoothing, για λέξεις
            # του V που δεν εμφανίζονται στο test set.
            log_w = float(np.log((count+1)/(len(words_c[c][0])+len(v))))
            log_llh[c].append(log_w)
    # Μετατροπή της δομής λεξικού σε DataFrame
    loglikelihood = pd.DataFrame(log_llh)
    
    return(voc, logprior, loglikelihood)


# Κληση του Αλγοριθμου
(voc, logprior, loglikelihood) = multinomialNB(inputDF, binaryNB=binNB)


#*****************************************************************************
#        ΠΡΟΒΛΕΨΗ ΣΕ ΑΓΝΩΣΤΟ ΚΕΙΜΕΝΟ
#*****************************************************************************
# Τα κείμενα προς πρόβλεψη μπορεί να είναι περισσότερα από ένα και είναι
# αποθηκευμένα στο DataFrame testDF


def predictDF(tdf):
    '''
    Μέθοδος υλοποίησης προβλέψεων σε άγνωστα κείμενα. Χρησιμοποιούνται οι πιθανότητες
    που υπολογίστηκαν κατά την εκπαίδευση του μοντέλου.  Για κάθε έγγραφο και για
    κάθε κατηγορία υπολογίζεται ένα σκορ το οποίο είναι ανάλογο με την πιθανότητα
    να ανήκει τελικά το έγγραφο στη συγκεκριμένη κατηγορία.
    
    \nΠαράμετροι:
     - tdf: Τα δεδομένα ελέγχου (κείμενο) σε δομή DataFrame.
     
    \n\nΕπιστρέφει:
   - Οι υπολογιζόμενες προβλέψεις για κάθε έγγραφο ελέγχου και για κάθε κατηγορία.
    
   :Example:
   >>> predictDF(testDataDF)
   >>> 
   '''

    # Αρχική αποθήκευση δεδομένων σε δομή λεξικού
    pred1 = {}
    for c in classes:
        pred1[c]= []

    # Για κάθε έγγραφου ελέγχου του DataFrame
    for i in range(len(tdf)):
        # Για κάθε κλάση
        for c in classes:
            # Η πιθανότητα εμφάνισης (logprior) της κλάσης
            score = logprior[c][0]
            # Για κάθε λέξη wi του εγγράφου προσθέτω στο score την log(P(wi,c)).
            # Λέξεις που δεν υπάρχουν στο V αφαιρούνται
            for w in tdf['Sentences'][i]:
                # Αν η λέξη δεν υπάρχει στο training set
                if w not in voc:
                    continue
                else:
                    # Αν υπάρχει προσθέτω τη loglikelihood πιθανότητα στο score
                    # Βρίσκω το index της λέξης στο DataFrame. Επιστρέφει ΛΙΣΤΑ
                    # με όλες τις πιθανές θέσεις. Επιλέγω την πρώτη.
                    index = loglikelihood.index[loglikelihood['V'] == w][0]
                    score += loglikelihood[c][index]
            # Αποθήκευση του score του εγγράφου στην κατηγορία
            pred1[c].append(score)
    # Μετατροπή των αποτελεσμάτων σε DataFrame και επιστροφή τους
    return(pd.DataFrame(pred1))

# Δομή αποθήκευσης των προβλέψεων των κειμένων ελέγχου
pred = predictDF(testDF)

print('\nΕκτιμήσεις για τα έγγραφα ελέγχου:\n')
print(pred)

