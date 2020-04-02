# =============================================================================
# HOMEWORK 5 - NEURAL NETWORKS
# MULTI-LAYER PERCEPTRON ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'neural_network' package, for creating and using a Multi_layer Perceptron classfier
# 'preprocessing' package, for rescaling ('normalizing') our data
from sklearn import 



# Load breast cancer dataset.
myData =


# Store features and target variable into 'X' and 'y'.
X = 
y =



# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fit) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0)




# Neural Networks are sensitive to the magnitudes of the features' values. Since we already
# split our dataset into 'train' and 'test' sets, we must rescale them separately (but with the same scaler)
# So, we rescale train data to the [0,1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
# Note that, test data should be transformed ONLY (not fit+transformed), since we require the exact
# same scaler used for the train data. 
# =============================================================================

minMaxScaler = 
x_train = 
x_test =


# =============================================================================




# It's time to create our classfier- 'MLPClassifier'. There are several hypermarameters
# that one could adjust in this case, such as:
# hidden_layer_sizes: A tuple, the length of which determines the number of hidden layers
#                     in the network, and the value at each hidden layers corresponds to
#                     the number of hidden units (neurons) within that layer.
#                     (e.g. hidden_layer_sizes = (10,10,8) means 3 hidden layers-
#                     1st layer with 10 neurons, 2nd layer with 10 neurons, and 3rd
#                     layer with 8 neurons).
# activation: This is the activation function used for the neural network. Can be
#             'identity', 'logistic', 'tanh' or 'relu'. Most common activation function
#             for deep neural networks is 'relu'.
# solver: This is the solver used for weight optimization. Available solvers in sklearn
#         are 'lbfgs' (Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm),
#         'sgd' (Stochastic gradient descent) and 'adam' (just adam). Default is 'adam', which
#         is better for large datasets, whereas 'lbfgs' converges faster and performs better
#         on small datasets.
# max_iter: Maximum number of iterations that shall determine that end of the algorithm if
#           no convergence (i.e. 'tolerance') has been reached until that point.  
# tol: A small number that determines whether the loss function has converged to a
#            point at which the iterations can terminate (tolerance).
# Unfortunately, the default and only loss function supported by sklearn is Cross-Entropy, so
# we cannot make this type of adjustment.
# =============================================================================


# ADD COMMAND TO CREAETE MLPCLASSIFIER HERE
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


# ADD COMMANDS TO COMPUTE METRICS HERE
print()
print()
print()

# =============================================================================