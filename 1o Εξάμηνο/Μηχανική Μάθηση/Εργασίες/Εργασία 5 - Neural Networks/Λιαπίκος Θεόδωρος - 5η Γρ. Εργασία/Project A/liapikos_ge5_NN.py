# =============================================================================
# HOMEWORK 5 - NEURAL NETWORKS
# MULTI-LAYER PERCEPTRON ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Οι χρησιμοποιούμενες βιβλιοθήκες
from sklearn import datasets, model_selection, metrics, neural_network, preprocessing

# Αποθήκευση της δομής του προς μελέτη DataSet
myData = datasets.load_breast_cancer()

# Οι ενεξάρτητες και η εξαρτημένη μεταβλητή
X = myData.data
y = myData.target

feature_names = myData.feature_names
target_names = myData.target_names

# Διαχωρισμός των δεδομένων σε σύνολα εκπαίδευσης και ελέγχου
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 42)

# Κανονικοποίηση των τιμών των ανεξάρτητων μεταβλητών στην κλίμακα [0,1]
minMaxScaler = preprocessing.MinMaxScaler(copy = False)
x_train = minMaxScaler.fit_transform(x_train)
x_test = minMaxScaler.fit_transform(x_test)


# Λίστα υπερπαραμέτρων για την αυτόματη δήλωση του μοντέλου
hyperparameters = [((10,),'sgd',0.0001), ((20,),'sgd',0.0001), ((20,),'adam',0.00001),
 ((50,50,50),'adam',0.00001), ((50,),'lbfgs',0.00001), ((100,100,100),'lbfgs',0.00001)]

# Δομή αποθήκευσης των συνολικών αποτελεσμάτων
results_total= []
for hls,solver,tolerance in hyperparameters:
    # Δομή αποθήκευσης των αποτελεσμάτων κάθε επιμέρους μοντέλου
    results = []
    results.append(hls)
    results.append(solver)
    results.append(tolerance)
    model = neural_network.MLPClassifier(hidden_layer_sizes=hls , activation='relu',
                    solver=solver, max_iter=100, tol=tolerance, random_state=42)

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
print('Hid.LayerSizes\t', 'Solver\t', 'Toler.\t', 'Recall\t', 'Prec\t', 'F1\t')
for result in results_total:
    print('{}\t\t {}\t {}\t {:.3f}\t {:.3f}\t {:.3f}\t'.format(result[0],
          result[1], result[2], result[3], result[4], result[5]))

