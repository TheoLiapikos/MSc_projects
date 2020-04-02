# =============================================================================
# HOMEWORK 2 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Εισαγωγή της χρησιμοποιύμενης βιβλιοθήκης
import Orange

# =============================================================================

# Εισαγωγή του μελετούμενου DataSet
wineData = Orange.data.Table('wine')


# Οι 3 εξεταζόμενοι learners της βιβλιοθήκης. Επιλογή ενός μόνο κάθε φορά.

# =============================================================================
#         1. Ordered Learner - Evaluator == Entropy
# =============================================================================

# Χρησιμοποιείται ο CN2Learner(), που χρησιμοποιεί εξ' ορισμού Entropy evaluator
learner = Orange.classification.rules.CN2Learner()


# =============================================================================
#        2.  Unordered Learner - Evaluator == Entropy
# =============================================================================

# Χρησιμοποιείται ο CN2UnorderedLearner(), που χρησιμοποιεί εξ' ορισμού τον 
# Laplace evaluator, σύμφωνα με την τεκμηρίωση της Orange.
learner = Orange.classification.rules.CN2UnorderedLearner()

# Δήλωση evaluator == Entropy
learner.rule_finder.quality_evaluator = Orange.classification.rules.EntropyEvaluator()


# =============================================================================
#         3. Ordered Learner - Evaluator == Laplace
# =============================================================================

# Χρησιμοποιείται ο CN2Learner(), που χρησιμοποιεί εξ' ορισμού Entropy evaluator
learner = Orange.classification.rules.CN2Learner()

# Δήλωση evaluator == Laplace
learner.rule_finder.quality_evaluator = Orange.classification.rules.LaplaceAccuracyEvaluator()



# =============================================================================
# Ρύθμιση των εξεταζόμενων παραμέτρων 'beam_width', 'min_covered_examples' και
# 'max_rule_length', του κάθε learner, στις βέλτιστες τιμές τους.
# =============================================================================
# Παράμετρος: beam_width, Εύρος τιμών που μελετήθηκε: 3,6,8,10
learner.rule_finder.search_algorithm.beam_width = 3

# Παράμετρος: min_covered_examples, Εύρος τιμών που μελετήθηκε: 7,10,13,15
learner.rule_finder.general_validator.min_covered_examples = 15

# Παράμετρος: max_rule_length, Εύρος τιμών που μελετήθηκε: 2,3,4,5
learner.rule_finder.general_validator.max_rule_length = 2



# =============================================================================
# Έλεγχος του μοντέλου με κλήση της CrossValidation() συνάρτησης της βιβλιοθήκης,
# δηλώνοντας το DataSet, τον learner, το πλήθος κατάτμησης k του DataSet, καθώς 
# και προσδιοριστικό του τρόπου τυχαιοποίησης των δεδομένων.
results = Orange.evaluation.testing.CrossValidation(wineData, [learner], k=10,
                                                    random_state=42)

# =============================================================================
# Έλεγχος της απόδοσης του κάθε μοντέλου
# Σύμφωνα με τις οδηγίες, οι μετρικές θα είναι οι: Precision, Recall και F1
print(Orange.evaluation.scoring.Precision(results, average='macro'))
print(Orange.evaluation.scoring.Recall(results, average='macro'))
print(Orange.evaluation.scoring.F1(results, average='macro'))

# =============================================================================
# Δημιουργία classifier με τροφοδότηση του DataSet στον learner. Με τον τρόπο 
# αυτό γίνεται η κατηγοριοποίηση και δημιουργούνται οι κανόνες που την υλοποιούν
classifier = learner(wineData)

# =============================================================================
# Εκτύπωση της λίστας των κανόνων (rules) του classifier 
for rule in classifier.rule_list:
    print(rule)
   
