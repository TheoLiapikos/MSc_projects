# =============================================================================
# HOMEWORK 4 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

from sklearn import datasets, model_selection, metrics, naive_bayes
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

textData = datasets.fetch_20newsgroups(subset='all')

X = textData.data
y = textData.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 42)

results_total= []

for NBalpha in np.arange(0.01,0.2,0.01):
    results = []
    results.append(NBalpha)
    model = make_pipeline(TfidfVectorizer(), naive_bayes.MultinomialNB(alpha = NBalpha))
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)

#    results.append(metrics.accuracy(y_test,y_predicted,average='macro'))
    results.append(metrics.recall_score(y_test,y_predicted,average='macro'))
    results.append(metrics.precision_score(y_test,y_predicted,average='macro'))
    results.append(metrics.f1_score(y_test,y_predicted,average='macro'))

    results_total.append(results)

print('alpha\t', 'Recall\t', 'Prec\t', 'F1\t')
for result in results_total:
    print('{:.3f}\t {:.3f}\t {:.3f}\t {:.3f}\t'.format
          (result[0], result[1], result[2], result[3]))
