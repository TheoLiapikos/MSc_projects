# =============================================================================
# HOMEWORK 6 - Support Vector Machines
# SUPPORT VECTOR MACHINE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Εισαγωγή των απαραίτητων βιβλιοθηκών
from sklearn import model_selection, metrics, svm, preprocessing
import pandas as pd

# Εισαγωγή των δεδομένων
myData = pd.read_csv('creditcard.csv')


# Απομόνωση των εξαρτημένων και των ανεξάρτητων μεταβλητών
X = myData.iloc[:, :-1].values
y = myData.iloc[:, -1:].values

# Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ελέγχου
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42, stratify=y)
y_train = y_train.ravel()


# Ποιοτικός έλεγχος - Κατανομή των δειγμάτων στις 2 κατηγορίες
y2 = pd.DataFrame(y)
for cat in (0,1):
    print('Πλήθος δεδομένων εκπαίδευσης στην κατηγορία', str(cat),': ', 
          int((y2.iloc[:,:][y2.iloc[:,:] == cat]).count()))


minMaxScaler = preprocessing.MinMaxScaler()
x_train = minMaxScaler.fit_transform(x_train)
x_test = minMaxScaler.transform(x_test)


# Λίστα υπερπαραμέτρων για την αυτόματη δήλωση του μοντέλου
hyperparameters = [
                   (0.1,'poly','auto',2), 
                   (10,'poly','auto',2),
                   (10,'poly','auto',3),
                   (10,'poly','auto',4),
                   (10,'poly','auto',5), 
                   (0.1,'rbf',0.3,3),
                   (10,'rbf',0.3,3),
                   (10,'rbf',5,3), 
                   (0.1,'sigmoid',0.5,3), 
                   (10,'sigmoid',2,3), 
                   (100,'sigmoid',5,3),
                   ]


results_total= []
for C,kernel,gamma,degree in hyperparameters:
    # Δομή αποθήκευσης των αποτελεσμάτων κάθε επιμέρους μοντέλου
    results = []
    results.append(C)
    results.append(kernel)
    results.append(gamma)
    results.append(degree)
#    class_weight='balanced', max_iter=1000
    model = svm.SVC(C=C,kernel=kernel,gamma=gamma,degree=degree,random_state=42)
    
    # Εκπαίδευση του μοντέλου
    model.fit(x_train, y_train)

    # Προβλέψεις επί του συνόλου ελέγχου
    y_predicted = model.predict(x_test)
    
    # Οι χρησιμοποιούμενες μετρικές
    recall = metrics.recall_score(y_test,y_predicted,average='macro')
    precision = metrics.precision_score(y_test,y_predicted,average='macro')
    f1 = metrics.f1_score(y_test,y_predicted,average='macro')

    results.append(recall)
    results.append(precision)
    results.append(f1)

    results_total.append(results)

# Εκτύπωση συνολικών αποτελεσμάτων
print('\n\nC\t', 'Kernel\t\t', 'Gamma\t', 'Degree\t', 'Recall\t', 'Prec\t', 'F1\t')
for result in results_total:
    if (result[1] == 'sigmoid'):
        print('{:.1f}\t {}\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t'.format(result[0],
              result[1], result[2], result[3], result[4], result[5], result[6]))
    else:
        print('{:.1f}\t {}\t\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t'.format(result[0],
              result[1], result[2], result[3], result[4], result[5], result[6]))

# =============================================================================
