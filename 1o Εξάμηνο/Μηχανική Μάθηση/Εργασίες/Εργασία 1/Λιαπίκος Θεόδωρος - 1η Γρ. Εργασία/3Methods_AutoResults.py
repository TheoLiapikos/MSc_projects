# Εισαγωγή των απαραίτητων βιβλιοθηκών για την εκτέλεση του προγράμματος
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
breastCancer = load_breast_cancer()


# =========================================================================== #
#####                  DECISION TREE                                      #####
# =========================================================================== #
rs=42
aver='binary'

results_total= []
for crit in ('gini', 'entropy'):
    for num_feat in (10,15,20,25,30):
        for max_dep in range(3, num_feat+1, 3):
            results = []
            results.append(crit)
            results.append(num_feat)
            results.append(max_dep)
            # Διαχωρισμός των Δεδομένων
            X = breastCancer.data[:, :num_feat]
            Y = breastCancer.target
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                                random_state=rs)
            # Δημιουργία μοντέλου
            tree_clf = DecisionTreeClassifier(criterion=crit, max_depth=max_dep,
                                               random_state=rs)
            # Εκπαίδεση του μοντέλου και προβλέψεις στο Test set
            tree_clf.fit(X_train, Y_train)
            Y_predicted = tree_clf.predict(X_test)
            # Ακρίβεια μοντέλου σε Training και Test sets
            results.append(tree_clf.score(X_train, Y_train))
            results.append(tree_clf.score(X_test, Y_test))
            # Προσθήκη επιπλέον μετρικών (recall,Precision, f1)
            results.append(metrics.recall_score(Y_test,Y_predicted,average=aver))
            results.append(metrics.precision_score(Y_test,Y_predicted,average=aver))
            results.append(metrics.f1_score(Y_test,Y_predicted,average=aver))
            # Προσθήκη απτελεσμάτων στα συνολικά
            results_total.append(results)

# Εκτύπωση των συνολικων αποτελεσμάτων
print('Criter.\t', 'feat\t', 'Depth\t', 'Tr_Acc\t', 'Te_Acc\t', 'Recall\t',
      'Prec.\t', 'f1')
for result in results_total:
    print('{}\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}'.format
          (result[0], result[1], result[2], result[3], result[4], result[5],
           result[6], result[7]))


# =========================================================================== #
#####                  RANDOM FOREST                                      #####
# =========================================================================== #
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

rs=42
aver='binary'

results_total= []
for crit in ('gini', 'entropy'):
    for num_feat in (10,15,20,25,30):
        for estim in (3,5,10,15,20):
            results = []
            results.append(crit)
            results.append(num_feat)
            results.append(estim)
            # Διαχωρισμός των Δεδομένων
            X = breastCancer.data[:, :num_feat]
            Y = breastCancer.target
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                                random_state=rs)
            # Δημιουργία μοντέλου
            rf_clf = RandomForestClassifier(criterion=crit, max_depth=None,
                                               random_state=rs, n_estimators=estim)
            # Εκπαίδεση του μοντέλου και προβλέψεις στο Test set
            rf_clf.fit(X_train, Y_train)
            Y_predicted = rf_clf.predict(X_test)
            # Ακρίβεια μοντέλου σε Training και Test sets
            results.append(rf_clf.score(X_train, Y_train))
            results.append(rf_clf.score(X_test, Y_test))
            # Προσθήκη επιπλέον μετρικών (recall,Precision, f1)
            results.append(metrics.recall_score(Y_test,Y_predicted,average=aver))
            results.append(metrics.precision_score(Y_test,Y_predicted,average=aver))
            results.append(metrics.f1_score(Y_test,Y_predicted,average=aver))
            # Προσθήκη απτελεσμάτων στα συνολικά
            results_total.append(results)

# Εκτύπωση των συνολικων αποτελεσμάτων
print('Criter.\t', 'feat\t', 'Estim\t', 'Tr_Acc\t', 'Te_Acc\t', 'Recall\t',
      'Prec.\t', 'f1')
for result in results_total:
    print('{}\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}'.format
          (result[0], result[1], result[2], result[3], result[4], result[5],
           result[6], result[7]))


# =========================================================================== #
#####               LOGISTIC REGRESSION                                   #####
# =========================================================================== #
from sklearn.linear_model import LogisticRegression

rs=42
aver='binary'

# Διαχωρισμός των διαφόρων sets δεδομένων
X = breastCancer.data
Y = breastCancer.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=rs)

results_total= []
for c in (0.001,0.01,1,5,10,25,50,100):
    results = []
    results.append(c)
    lr_clf = LogisticRegression(C=c, solver='liblinear', random_state=rs)
    
    lr_clf.fit(X_train, Y_train)
    Y_predicted = lr_clf.predict(X_test)
    
    results.append(lr_clf.score(X_train, Y_train))
    results.append(lr_clf.score(X_test, Y_test))
    
    results.append(metrics.recall_score(Y_test,Y_predicted,average=aver))
    results.append(metrics.precision_score(Y_test,Y_predicted,average=aver))
    results.append(metrics.f1_score(Y_test,Y_predicted,average=aver))
    
    results_total.append(results)

# Εκτύπωση των συνολικων αποτελεσμάτων
print('C_par.\t','Tr_Acc\t', 'Te_Acc\t', 'Recall\t','Prec.\t', 'f1')
for result in results_total:
    print('{}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t {:.3f}'.format(result[0],
          result[1],result[2],result[3],result[4],result[5]))

