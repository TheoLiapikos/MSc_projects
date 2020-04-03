# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import warnings
import pandas as pd
#from pandas.core import datetools
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from costcla.metrics import cost_loss
from sklearn.calibration import CalibratedClassifierCV
from costcla.models import BayesMinimumRiskClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
#from sklearn.model_selection import KFold

# Download from Internet
#path_cleveland = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
#path_hungary = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
#path_swiss = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data"
#path_veniceb = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data"

# Use locally stored files
path_cleveland = "data/processed.cleveland.data"
path_hungary = "data/processed.hungarian.data"
path_swiss = "data/processed.switzerland.data"
path_veniceb = "data/processed.va.data"


paths = [path_cleveland, path_hungary, path_swiss, path_veniceb]
columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", 
         "ca", "thal", "target"]

# Δομή αποθήκευσης των 4 επιμέρους DataFrames δεδομένων
dfs = []
for i in range(len(paths)):
    dfs.append(pd.read_csv(paths[i], names=columns))

# Συνένωση των DataFrames δεδομένων
data = pd.concat(dfs)

# Εναλλακτικός τρόπος σε μία γραμμή
#data = pd.concat(map(lambda x: pd.read_csv(x, names=columns), paths))

print(data.head())

#Dealing with missing data
# 1. Replace '?' symbol with 'nan'
data.replace("?", np.nan, inplace=True)
#print(data.isnull().sum())

# 2. Delete rows with missing data
data.dropna(axis=0, inplace=True)
data.reset_index(drop=True, inplace=True)

# Replace target labels 2,3, and 4 with 1
data['target'].replace(to_replace=[2, 3, 4], value=1, inplace=True)

# Distribution of target labels
for i in range(2):
    counts = data.target.value_counts().values[i]
    print('Label: %1d, Counts: %3d (%.2f%s)' %(i, counts, counts/data.shape[0],'%'))

# Χωρισμός μεταβλητών
X = data.drop('target', axis=1)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ίσως χρειαστεί να κάνω κανονικοποίηση των δεδομένων
#from sklearn import preprocessing
#X = preprocessing.normalize(data.drop('target'))

# Create CostMatrix based on informations taken from:
# http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
# 0 absence of heart disease, 1 presence of heart disease
# Misclassification costs
# Class 0 = 1,  Class 1 = 5
# Create data for fp, fn, tp, tn
fp = np.full((y_test.shape[0],1), 1)
fn = np.full((y_test.shape[0],1), 5)
tp = np.zeros((y_test.shape[0],1))
tn = np.zeros((y_test.shape[0],1))
cost_matrix = np.hstack((fp, fn, tp, tn))


print("random forest")
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
loss = cost_loss(y_test, y_pred, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, y_pred).T) # transpose to align with slides


print("\nno calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\ncostcla calibration on training set")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
model = clf.fit(X_train, y_train)
prob_train = model.predict_proba(X_train)
bmr = BayesMinimumRiskClassifier(calibration=True)
bmr.fit(y_train, prob_train) 
prob_test = model.predict_proba(X_test)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nsigmoid calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides

print("\nisotonic calibration")
clf = RandomForestClassifier(random_state=0, n_estimators=100)
cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
model = cc.fit(X_train, y_train)
prob_test = model.predict_proba(X_test)
bmr = BayesMinimumRiskClassifier(calibration=False)
pred_test = bmr.predict(prob_test, cost_matrix)
print(classification_report(y_test, pred_test))
loss = cost_loss(y_test, pred_test, cost_matrix)
print("%d\n" %loss)
print(confusion_matrix(y_test, pred_test).T) # transpose to align with slides




