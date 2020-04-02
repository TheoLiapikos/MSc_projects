# =============================================================================
# HOMEWORK 1 - DECISION TREES
# DECISION TREE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'datasets', for our data
# 'metrics' package, for measuring scores
# 'ensemble' package, for calling the Random Forest classifier
# 'model_selection', (instead of the 'cross_validation' package), which will help validate our results.
# =============================================================================


# IMPORT NECESSARY LIBRARIES HERE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import graphviz

# =============================================================================



# Load breastCancer data
# =============================================================================


# ADD COMMAND TO LOAD DATA HERE
from sklearn.datasets import load_breast_cancer
breastCancer = load_breast_cancer()



# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target



# RandomForestClassifier() is the core of this script. You can call it from
# You can customize its functionality in various ways, but for now simply play
# with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for
# the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but
# it will take longer to compute. Also, there is a critical number after which
# there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting,
# so start with a maxDepth of e.g. 3, and increase it slowly by evaluating the
# results each time.
# =============================================================================


# ADD COMMAND TO CREATE DECISION TREE CLASSIFIER MODEL HERE
model = RandomForestClassifier(criterion="gini", max_depth=None, random_state=10,
								n_estimators=5)


# =============================================================================



# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=10)



# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE
model.fit(x_train, y_train)

# =============================================================================


# Ok, now let's predict the output for the test input set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted = model.predict(x_test)


# =============================================================================



# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform
# a type of averaging on the data. DON'T WORRY ABOUT THAT JUST YET. USE EITHER 'MICRO' OR 'MACRO'.
# =============================================================================

from sklearn.metrics import classification_report,confusion_matrix



# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print('Confusion Matrix: \n', confusion_matrix(y_test, y_predicted))
print('Classification Report: \n', classification_report(y_test, y_predicted))

#comat = confusion_matrix(Y_test,Y_predicted)
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()

print('Επιτυχώς διαγνωσθέντες ασθενείς: %d' %tp)
print('Εσφαλμένα διαγνωσθέντες ασθενείς: %d' %fp)
print('Επιτυχώς διαγνωσθέντες φυσιολογικοί: %d' %tn)
print('Εσφαλμένα διαγνωσθέντες φυσιολογικοί: %d' %fn)

# =============================================================================
# Το RandomForest model που εκπαιδεύτηκε, παράγει ουσιαστικά μια λίστα από trees,
# αποθηκευμένη στο attribute estimators_, την οποία μπορούμε να την προσπελάσουμε
# χρησιμοποιώντας τα κατάλληλα indeces.
len(model.estimators_)
from sklearn import tree


dot_data = tree.export_graphviz(model.estimators_[0], out_file=None,
						feature_names = breastCancer.feature_names[:numberOfFeatures],
                         filled=True, rounded=True,
						 class_names = breastCancer.target_names,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render("breastCancerFirstTreeFromForestPlot", cleanup=True, format='png')
graph


