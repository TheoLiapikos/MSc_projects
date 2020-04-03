# ****************************************************************************
# N-Gram Model
# ****************************************************************************
# Μοντέλο που συσχετίζει κάθε πιθανή αλληλουχία Ν λέξεων ενός corpus με την
# πιθανότητα να ακολουθείται από κάποια άλλη συγκεκριμένη λέξη.
# Το μοντέλο χτίζεται διατρέχοντας το κείμενο και κάθε αλληλουχία Ν λέξεων που
# εντοπίζεται εισαγεται σε λεξικό μαζί με τη λέξη που την ακολουθεί.

# ΠΡΟΣΟΧΗ: Στο μεταπτυχιακό ορίσαμε σαν N-Gramm την πιθανότητα Ν-1 λέξεις να
# ακολουθούνται από κάποια άλλη συγκεκριμένη λέξη. Η τροποποίηση είναι εύκολη.

import random
import nltk

# Κείμενο δοκιμής
text = """Global warming or climate change has become a worldwide concern. It 
is gradually developing into an unprecedented environmental crisis 
evident in melting glaciers, changing weather patterns, rising sea levels, 
floods, cyclones and droughts. Global warming implies an increase in the 
average temperature of the Earth due to entrapment of greenhouse gases in 
the earth’s atmosphere."""

# Μέγεθος παραθύρου Ν
n = 2

# Λεξικό αποθήκευσης των N-Grams
ngrams = {}

# Χτίσιμο του μοντέλου.
# Το κείμενο χωρίζεται σε λέξεις και στη συνέχεια σαρώνεται σειριακά. Κάθε N-Gram
# που προκύπτει εισάγεται στο λεξικό μαζί με την λέξη που το ακολουθεί. Άρα στο
# τέλος κάθε N-Gram θα συνοδεύεται από όλες τις πιθανές λέξεις που το ακολουθούν.

words = nltk.word_tokenize(text)
for i in range(len(words)-n):
    gram = ' '.join(words[i:i+n])
    if gram not in ngrams.keys():
        ngrams[gram] = []
    ngrams[gram].append(words[i+n])

# Δοκιμή του μοντέλου
# Επειδή το κείμενο είναι απλό κάθε πιθανό N-Gram μπορεί να ακολουθείται από μία
# μόνο συγκεκριμένη λέξη. Έτσι ξεκινώντας από το πρώτο N-Gram βλέπουμε αν μπορεί
# να αναπαραχθεί το αρχικό κείμενο σε κάποιο βάθος.
currentGram = ' '.join(words[0:n])
result = currentGram
for i in range(50):
    if currentGram not in ngrams.keys():
        break
    possibilities = ngrams[currentGram]
    nextItem = possibilities[random.randrange(len(possibilities))]
    result += ' '+nextItem
    rWords = nltk.word_tokenize(result)
    currentGram = ' '.join(rWords[len(rWords)-n:len(rWords)])

print(result)