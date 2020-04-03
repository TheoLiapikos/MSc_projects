#!/usr/bin/env python3

#*****************************************************************************
#        Υλοποίηση αλγορίθμου Multivariate Bernoulli Naive Bayes
# 
# Η υλοποίηση, κυρίως το κομμάτι των αναγκαίων μαθηματκών υπολογισμών, βασίστηκε
# στα στοιχεία που δίνονται στο paper 'Spam Filtering with Naive Bayes – Which
# Naive Bayes?'. 
# Μέθοδοι που είναι κοινοί με υπόλοιπους αλγορίθμους, βρίσκονται αποθηκευμένοι
# στο αρχείο 'commonModules.py', απ' όπου και εισάγονται.
#
#*****************************************************************************

import pandas as pd
import numpy as np
from commonModules import preprocDF

# Ορισμός των υπερπαραμέτρων του Αλγορίθμου
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

# Συνολικό πλήθος δεδομένων (προτάσεων) εισόδου
ndoc = len(inputDF)

# Το πλήθος των training docs κάθε κλάσης
mc = pd.DataFrame()
for c in classes:
    mc[c] = [inputDF['Labels'][inputDF['Labels'] == c].count()]

# Η πιθανότητα εμφάνισης (logprior) κάθε κλάσης
logprior = pd.DataFrame()

# Το συνολικό λεξιλόγιο του corpus (μοναδικές λέξεις)
voc = []

# Η παράμετρος Μ(t,c) δείχνει πόσα docs του training set της κλάσης c περιέχουν
# το token (λέξη) t 
mtc = pd.DataFrame()

# Η παράμετρος P(t,c)=1+M(t,c)/2+Mc υπολογίζεται για κάθε token σε κάθε κλάση
ptc = pd.DataFrame()


def bernoulliNB(inputData):
    '''
    Μέθοδος υλοποίησης της φάσης εκπαίδευσης του αλγορίθμου Multivariate Bernoulli
    Naive Bayes.Τα δεδομένα εισόδου (επισημασμένα κείμενα) χρησιμοποιούνται για
    την εκπαίδευση του μοντέλου. Υπολογίζονται και επιστρέφονται τα απαραίτητα
    δεδομένα για την πραγματοποίηση προβλέψεων στα ακόλουθα στάδια του αλγορίθμου.
    
    \nΠαράμετροι:
     - inputData: Τα δεδομένα εισόδου (κείμενα) σε δομή DataFrame.
     
    \nΕπιστρέφει:
    - Το λεξιλόγιο που παράγεται από τα έγγραφα εκπαίδευσης.
    - Η παράμετρος Μ(t,c) δείχνει πόσα έγγραφα εκπαίδευσης της κλάσης c περιέχουν
    το token (λέξη) t
    - Η παράμετρος P(t,c)=1+M(t,c)/2+Mc, που υπολογίζεται για κάθε token σε κάθε
    κλάση. Η μεταβλητή Mc αντιστοιχεί στο πλήθος εγγράφων εκπαίδευσης της κατηγορίας c.
           
    :Example:
    >>> bernoulliNB(dataDF)
    >>> 
    '''

    # Υπολογισμός απαραίτητων μεγεθών του μοντέλου

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

    # Η παράμετρος Μ(t,c) δείχνει πόσα docs του training set της κλάσης c περιέχουν
    # το token (λέξη) t 
    mtc = pd.DataFrame()
    for c in classes:
        results = []
        for i in range(len(voc)):
            count = 0
            for doc in inputData['Sentences'][inputData['Labels'] == c]:
                if voc[i] in doc:
                    count+=1
            results.append(count)
        mtc[c] = [results]

    # Η παράμετρος P(t,c)=1+M(t,c)/2+Mc υπολογίζεται για κάθε token σε κάθε κλάση
    ptc = pd.DataFrame()
    for c in classes:
        results = []
        for i in range(len(voc)):
            m_t_c = mtc[c][0][i]
            m_c = mc[c]
            p_t_c = (1+m_t_c)/(2+m_c)
            results.append(p_t_c)
        ptc[c] = [results]
        
    return(voc, logprior, mtc, ptc)


# Κληση του Αλγοριθμου
(voc, logprior, mtc, ptc) = bernoulliNB(inputDF)


#*****************************************************************************
#        ΠΡΟΒΛΕΨΗ ΣΕ ΑΓΝΩΣΤΟ ΚΕΙΜΕΝΟ
#*****************************************************************************
# Τα κείμενα προς πρόβλεψη μπορεί να είναι περισσότερα από ένα και είναι
# αποθηκευμένα στο DataFrame testDF

# Δομή αποθήκευσης των vectors των προς πρόβλεψη κειμένων
pred = pd.DataFrame(columns=['Vectors'])


# Μετατροπή ενός tokenized κειμένου σε vector και αποθήκευσή του
def vectorizeDoc(doc, index):
    '''
    Μέθοδος μετατροπής ενός εγγράφου ελέγχου σε διάνυσμα (vector).
    Το παραγόμενο διάνυσμα έχει μέγεθος ίσο με το λεξιλόγιο που παράγουν τα
    έγγραφα εκπαίδευσης του μοντέλου.
    
    Οι τιμές του διανύσματος είναι 1 ή 0
    ανάλογα αν η λέξη της συγκεκριμένης θέσης του εγγράφου υπάρχει ή όχι στο
    λεξιλόγιο. Με τον τρόπο αυτό λέξεις οι οποίες εμφανίζονται μόνο στα έγγραφα
    ελέγχου και όχι στα έγγραφα εκπαίδευσης αγνοούνται.
    
    \nΠαράμετροι:
    - doc: Το προς μετατροπή έγγραφο.
    - index: Το index του εγγράφου στα έγγραφα ελέγχου.
            
    :Example:
    >>> vectorizeDoc(document, 6)
    >>> 
    '''
    doc_v = set(doc)
    # Δομή χτισίματος του vector του κειμένου
    vec = []
    # Σάρωση όλου του λεξιλογίου v του corpus. Αν μια λέξη του λεξιλογίου v υπάρχει
    # και στο εξεταζόμενο κείμενο θέτω 1 στην αντίστοιχη θέση του vector, αλλιώς 0.
    # Άγνωστες λέξεις που δεν υπάρχουν στο v με αυτόν τον τρόπο ΑΓΝΟΟΥΝΤΑΙ
    for j in range(len(voc)):
        if voc[j] in doc_v:
            vec.append(1)
        else:
            vec.append(0)
    # Αποθήκευση του τελικού vector στην αντίστοιχη δομή
    pred.loc[index] = [vec]


def predictDF(tdf):
    '''
    Μέθοδος υλοποίησης προβλέψεων σε άγνωστα κείμενα. Χρησιμοποιούνται οι πιθανότητες
    που υπολογίστηκαν κατά την εκπαίδευση του μοντέλου. Για κάθε έγγραφο και για
    κάθε κατηγορία υπολογίζεται ένα σκορ το οποίο είναι ανάλογο με την πιθανότητα
    να ανήκει τελικά το έγγραφο στη συγκεκριμένη κατηγορία.
    
    Ο υπολογισμός του σκορ κάθε εγγράφου γίνεται με τον πολύπλοκο τύπο που 
    εμφανίζεται στο τέλος της 2ης σελίδας του paper 'Spam Filtering with Naive
    Bayes – Which Naive Bayes?', που μας χορηγήθηκε ως επιπλέον υλικό μελέτης.
        
    Απαραίτητη προϋπόθεση για την λειτουργία της μεθόδου και τον υπολογισμό των
    απαραίτητων μεγεθών, είναι το εξεταζόμενο έγγραφο να μετατραπεί σε αναπαράσταση
    διανύσματος (vector) με την κλήση της κατάλληλης μεθόδου.
    
    \nΠαράμετροι:
     - tdf: Τα δεδομένα ελέγχου (κείμενα) σε δομή DataFrame.
        
    :Example:
    >>> predictDF(testDataDF)
    >>> 
    '''

    global pred
    
    # Βολική δομή Λεξικού, όπου αποθηκεύονται τα συνολικά στοιχεία κάθε κλάσης
    # πριν μετατραπεί σε στήλη ενός DataFrame
    temp = {}
#    temp['Vectors'] = []
    for c in classes:
        temp[str(c)] = []
    
    # Για κάθε έγγραφο ελέγχου του DataFrame
    for i in range(len(tdf)):
        # Μετατροπή του εγγράφου σε vector
        vectorizeDoc(tdf['Sentences'][i], i)
        # Για κάθε κλάση
        for c in classes:
            score = logprior[c][0]
            # Σάρωση κάθε xi του vector του κειμένου
            for j in range(len(pred['Vectors'][0])):
                xi = pred['Vectors'][i][j]
                score += float(np.log(np.power(ptc[c][0][j],xi)*np.power(1-ptc[c][0][j],1-xi)))
            temp[str(c)].append(score)
    # Μετατροπή των αποτελεσμάτων σε DataFrame και συνέννωση με δομή εγγράφων ελέγχου
    temp = pd.DataFrame(temp)
    pred = pd.concat([pred, temp], axis=1)


predictDF(testDF)

print('\nΕκτιμήσεις για τα έγγραφα ελέγχου:\n')
print(pred.loc[:,'0':'1'])

