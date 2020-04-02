
# Εισαγωγή των απαραίτητων βιβλιοθηκών για την εκτέλεση του προγράμματος
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
import graphviz

# Τα προς επεξεργασία δεδομένα βρίσκονται έτοιμα στις βιβλιοθήκες με datasets
# που προσφέρει το sklearn
from sklearn.datasets import load_breast_cancer
breastCancer = load_breast_cancer()

rs = 42 # random_state

# =========================================================================== #
#####                  DECISION TREE                                      #####
# =========================================================================== #
crit="entropy"   # Criterion. "gini" or "entropy"
nof = 20      # Number of Features
max_dep=6    # max_depth

# Από την μελέτη των περιεχομένων του dataset φαίνεται ότι αναφέρεται σε συνολικά
# 569 περιστατικά γυναικών, για κάθε μία από τις οποίες υπάρχουν καταγεγραμμένα
# συνολικά 30 χαρακτηριστικά.
# Διαχωρισμός των δεδομένων σε ανεξάρτητες και εξαρτημένες μεταβλητές. Το dataset
# είναι της μορφής dictionary με τα δεδομένα των δεδομένα των ανεξαρτήτων μεταβλητών
# να βρίσκονται στο κλειδί data και τα δεδομένα της ανεξάρτητης μεταβλητής στο
# κλειδί target
X = breastCancer.data[:, :nof]
Y = breastCancer.target

# ΔΗΜΙΟΥΡΓΙΑ ΤΟΥ CLASSIFIER
# ΠΡΟΣΟΧΗ!!! Πρέπει να δηλωθεί random_state και εδώ αλλά και παρακάτω στην split
# εντολή, ώστε να έχω αναπαραγώγιμα αποτελέσματα
tree_clf = DecisionTreeClassifier(criterion=crit, max_depth=max_dep, random_state=rs)

 # Διαχωρισμός των δεδομένων σε σύνολα Εκπαίδευσης (train), με το 75% των περιστατικών
# (426) και Ελέγχου (test), με το 25% των περιστατικών (143)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=rs)

# Εκπαίδευση του μοντέλου με τα δεδομένα Εκπαίδευσης
tree_clf.fit(X_train,Y_train)

# Γραφική απεικόνιση του παραγόμενου Δέντρου Απόφασης
# Γίνεται χρήση της βιβλιοθήκης export_graphviz του module tree.
# Παράλληλα δηλώνονται τα ονόματα των χαρκτηριστικών (features) και των επιμέρους
# κατηγοριών (classes)
dot_data = tree.export_graphviz(tree_clf, out_file=None,
                        feature_names = breastCancer.feature_names[:nof],
                         filled=True, rounded=True,
                         class_names = breastCancer.target_names,
                         special_characters=True)

# Η παρακάτω εντολή εξάγει το δέντρο σε μορφή εικόνας (.png).
graph = graphviz.Source(dot_data)
graph.render("TreePlot_"+crit, cleanup=True, format='png')
graph


# =========================================================================== #
#####                  RANDOM FOREST                                      #####
# =========================================================================== #
crit="entropy"   # Criterion. "gini" or "entropy"
nof = 25      # Number of Features
estim=10    # Number of Estimators

X = breastCancer.data[:, :nof]
y = breastCancer.target


rf_clf = RandomForestClassifier(criterion=crit, max_depth=None, random_state=rs,
								n_estimators=estim)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=rs)


rf_clf.fit(x_train, y_train)

dot_data = tree.export_graphviz(rf_clf.estimators_[0], out_file=None,
						feature_names = breastCancer.feature_names[:nof],
                         filled=True, rounded=True,
						 class_names = breastCancer.target_names,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("RandomForestPlot_"+crit, cleanup=True, format='png')
graph

