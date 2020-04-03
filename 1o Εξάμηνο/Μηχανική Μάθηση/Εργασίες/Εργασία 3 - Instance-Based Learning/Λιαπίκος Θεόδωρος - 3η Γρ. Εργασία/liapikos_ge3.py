# =============================================================================
# HOMEWORK 3 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# Εισαγωγή απαραίτητων βιβλιοθηκών
from sklearn import model_selection, metrics, preprocessing, neighbors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Διάβασμα των δεδομένων από το .csv αρχείο
# Πρέπει να του δώσω ολόκληρο το path αλλιώς δεν το διαβάζει
diabetesData = pd.read_csv('/media/Personal Files/Theo Files/MSc/4. Μηχανική Μάθηση/Εργασίες/Εργασία 3 - Instance-Based Learning/Λιαπίκος Θεόδωρος - 3η Γρ. Εργασία/diabetes.csv')

# Διαμοιρασμός των δεδομένων στις ανεξάρτητες και την εξαρτημένη μεταβλητή
X = diabetesData.iloc[:,:-1]
y = diabetesData.iloc[:,-1:]

# Απαραίτητη για την ορθή λειτουργία του αλγορίθμου η κανονικοποίηση των τιμών των μεταβλητών
minMaxScaler = preprocessing.MinMaxScaler()
X_rescaled = minMaxScaler.fit_transform(X)

# Διαχωρισμός των δεδομένων σε σετ εκπαίδευσης-ελέγχου σε ποσοστό 75%-25%
# Η παράμετρος stratify=y διατηρεί την αναλογία των κλάσεων που υπήρχε στο αρχικό
# σύνολο δεδομένων, και στα σύνολα training και test. Έτσι το ποσοστό της κλάσης
# 1 είναι 0,349 τόσο στο y όσο και στα y_train και y_test.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X_rescaled, y, stratify = y, random_state = 42)

# Δομές αποθήκευσης των τιμών των μετρικών
k = 200
rec = np.arange(0,k, dtype = np.float64)
prec = np.arange(0,k, dtype = np.float64)
f1 = np.arange(0,k, dtype = np.float64)

# Επαναληπτική εκτέλεση του αλγορίθμου για διάφορες τιμές παραμέτρων του και για
# πλήθος γειτόνων από 1 έως 200
for weight in ('uniform','distance'):
    for ps in (1,2,3):
        for n in range(1,k):
            # =============================================================================
        
            # ADD COMMAND TO CREATE KNEIGHBORSCLASSFIER HERE
            model = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weight, p=ps, metric='minkowski')
            
            # =============================================================================
            
            # ADD CODE TO TRAIN YOUR MODEL HERE
            # Απαραίτητη η χρήση της μεθόδου values.ravel() για να μη χτυπάει ο αλγόριθμος
            model.fit(x_train, y_train.values.ravel())
            
            # =============================================================================

            # ADD COMMAND TO MAKE A PREDICTION HERE
            y_predicted = model.predict(x_test)
            
            # =============================================================================
            
            # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
            rec[n] = metrics.recall_score(y_test,y_predicted,average='macro')
            prec[n] = metrics.precision_score(y_test,y_predicted,average='macro')
            f1[n] = metrics.f1_score(y_test,y_predicted,average='macro')
            
            # =============================================================================

        # Αξιολόγηση αλγορίθμου για κάθε σετ τιμών των παραμέτρων
        # Το καλύτερο σκορ της F1 και το αντίστοιχο index της
        bF1 = f1.max()
        bF1_index = f1.argmax()
        print('\n\n')
        print('Best F1 score: {:.3f}'.format(f1.max()))
        print('Precision at best F1 score index: {:.3f}'.format(prec[bF1_index]))
        print('Recall at best F1 score index: {:.3f}'.format(rec[bF1_index]))
        print('Neighbors count at best F1 score: {}'.format(bF1_index+1))
        
        # =============================================================================
        
        # Εκτύπωση των αποτελεσμάτων για κάθε σετ τιμών των παραμέτρων
        # ADD COMMANDS TO PLOT RESULTS HERE
        xs = np.arange(1,k+1)
        plt.figure(1)
        plt.plot(xs, rec, label="Recall")
        plt.plot(xs, prec, label="Precision")
        plt.plot(xs, f1, label="F1-score")
        plt.xlabel('Neighbors')
        plt.ylabel('Metrics\' score')
        plt.legend()
        plt.annotate('weight: {}\nmetric\'s p: {}\nBest F1: {:.3f}\nNeighbors: {}'
                     .format(weight, ps, bF1, bF1_index+1), xy=(25,0.0), xytext=(25,0.0),)
        plt.show()

# =============================================================================
