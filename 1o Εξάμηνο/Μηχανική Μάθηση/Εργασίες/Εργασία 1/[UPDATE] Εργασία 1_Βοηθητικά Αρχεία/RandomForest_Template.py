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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


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
 model =


# =============================================================================



# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y)



# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN YOUR MODEL HERE


# =============================================================================




# Ok, now let's predict the output for the test input set
# =============================================================================


# ADD COMMAND TO MAKE A PREDICTION HERE
y_predicted =


# =============================================================================



# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform
# a type of averaging on the data. DON'T WORRY ABOUT THAT JUST YET. USE EITHER 'MICRO' OR 'MACRO'.
# =============================================================================



# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print()
print()
print()


# =============================================================================