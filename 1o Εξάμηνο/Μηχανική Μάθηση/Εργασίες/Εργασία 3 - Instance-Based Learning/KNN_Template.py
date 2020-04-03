# =============================================================================
# HOMEWORK 3 - INSTANCE-BASED LEARNING
# K-NEAREST NEIGHBORS TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# From sklearn, we will import:
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'preprocessing' package, for rescaling ('normalizing') our data
# 'neighbors' package, for creating and using KNN classfier
from sklearn import

# We also need 'pandas' and 'numpy' libraries
# for manipulating data.
import
import


# 'matplotlib.pyplot' is ncessary as well, for plotting results.
import



# Import data from file. Pandas will read this file as a 'dataframe' object (using 'read_csv' command)
# so we will treat that object accordingly.
# (FYI, a dataframe object has more infromation than just values, but we'll
# stick just to values for this project)
diabetesData =


# From diabetesData object, read included features and target variable.
# You can select a specific row/column of the dataframe object by using 'iloc',
# as if you were accessing an array (tip: you can access the last row or column
# with 'iloc' using the '-1' index).
# Make sure that, after selecting (/accessing) the desired rows and columns,
# you extract their 'values', which is what you want to store in variables 'X' and 'y'.
X = 
y = 


# KNN is sensitive to the magnitudes of the features' values, so we must rescale
# data to the [0, 1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
minMaxScaler = 
X_rescaled = 


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# The 'stratify' parameter will split the dataset proportionally to the target variable's values.
# Also, 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = 


# Initialize variables for number of neighbors, and arrays for holding values
# for recall, precision and F1 scores.
k = 200
rec = np.arange(1,k, dtype = np.float64)
prec = np.arange(1,k, dtype = np.float64)
f1 = np.arange(1,k, dtype = np.float64)

# Run the classification algorithm 'k' times (k = number of neighbors)
for n in range(1,k):

    
    # ΚNeighborsClassifier is the core of this script. You can customize its functionality
    # in various ways- for this project, just tweak the following parameters:
    # 'n_neighbors': The number of neighbors to use.
    # 'weights': Can be either 'uniform' (i.e. all points have equal weights) or 'distance' (points are
    #            assigned weights according to their distance from the query point).
    # 'metric': The metric used for measuring the distance between points. Could 'euclidean',
    #           'manhattan', 'minkowski', etc. Keep in mind that you need to tweak the 
    #           power parameter 'p' as well when using the 'minkowski' distance.
    # =============================================================================

    # ADD COMMAND TO CREATE KNEIGHBORSCLASSFIER HERE
    model = 
    
    # =============================================================================

    
    
    # Let's train our model.
    # =============================================================================
    
    
    # ADD CODE TO TRAIN YOUR MODEL HERE
    
    
    # =============================================================================



    
    # Ok, now let's predict the output for the second subset
    # =============================================================================
    
    
    # ADD COMMAND TO MAKE A PREDICTION HERE
    y_predicted = 
    
    
    # =============================================================================
    
    
    
    # Time to measure scores. We will compare predicted output (from input of x_test)
    # with the true output (i.e. y_test).
    # You can call 'recall_score()', 'precision_score()', 'f1_score()' or any other available metric
    # from the 'metrics' library.
    # The 'average' parameter is used while measuring metric scores to perform 
    # a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET. USE EITHER 
    # 'MICRO' OR 'MACRO' (preferably 'macro' for final results).
    # =============================================================================
    
    
    # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
    
    
    # =============================================================================
        

# Get best value of F1 score, and its index
best_F1 = 
bestF1_ind = 


# Plot stored results
# =============================================================================

# ADD COMMANDS TO PLOT RESULTS HERE

# =============================================================================
