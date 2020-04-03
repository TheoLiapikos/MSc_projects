# =============================================================================
# HOMEWORK 4 - BAYESIAN LEARNING
# NAIVE BAYES ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'naive_bayes' package, for creating and using Naive Bayes classfier
from sklearn import

# We also need to import 'make_pipeline' from the 'pipeline' module.
from __ import 

# We are working with text, so we need an appropriate package
# that shall vectorize words within our texts.
# 'TfidfVectorizer' from 'feature_extraction.text'.
from __ import 

# 'matplotlib.pyplot' and 'seaborn' are ncessary as well,
# for plotting confusion matrix.
import matplotlib.pyplot as plt
import seaborn as sns;


# Load text data.
textData = 


# Store features and target variable into 'X' and 'y'.
X = textData.data
y = textData.target


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0)



# We need to perform a transformation on the model that will later become 
# our Naive Bayes classifier. This transformation is text vectorization,
# using TfidfVectorizer().
# When you want to apply several transfromations on a model, and an
# estimator at the end, you can use a 'pipeline'. This allows you to
# define a chain of transformations on your model, like a workflow.
# In this case, we have one transformer that we wish to apply (TfidfVectorizer)
# and an estimator afterwards (Multinomial Naive Bayes classifier).
# =============================================================================


# ADD COMMAND TO MAKE PIPELINE HERE
alpha = 0.1 # This is the smoothing parameter for Laplace/Lidstone smoothing
model = 


# =============================================================================




    
# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN MODEL HERE 


# =============================================================================




# Ok, now let's predict output for the second subset
# =============================================================================


# ADD COMMAND TO MAKE PREDICTION HERE
y_predicted = 


# =============================================================================





# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform 
# a type of averaging on the data. Use 'macro' for final results.
# =============================================================================


# ADD COMMANDS TO COMPUTE METRICS HERE (AND PRINT ON CONSOLE)
recall = 
precision = 
f1 = 

print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)


# =============================================================================




# In order to plot the 'confusion_matrix', first grab it from the 'metrics' module
# and then throw it within the 'heatmap' method from the 'seaborn' module.
# =============================================================================

# ADD COMMANDS TO PLOT CONFUSION MATRIX
confusionMatrix = 



plt.title('Multinomial NB - Confusion matrix (a = %.2f) [Prec = %.5f, Rec = %.5f, F1 = %.5f]' % (alpha, precision, recall, f1))
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.show()


# =============================================================================
