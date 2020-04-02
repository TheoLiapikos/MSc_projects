# =============================================================================
# HOMEWORK 7 - CLUSTERING
# CLUSTERING ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'cluster' package, for importing the corresponding clustering algorithm
# 'preprocessing' package, for rescaling ('normalizing') our data
from sklearn import 



# Load a dataset.
myData =



# Most clustering methods are sensitive to the magnitudes of the features' values. Since we already
# split our dataset into 'train' and 'test' sets, we must rescale them separately (but with the same scaler)
# So, we rescale train data to the [0,1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
# Note that, test data should be transformed ONLY (not fit+transformed), since we require the exact
# same scaler used for the train data. 
# =============================================================================

minMaxScaler = 
myData =


=============================================================================




# It's time to create our clustering algorithm. Most algorithms share many common
# hyperparamters, but depending on their nature they tend to be tuned by different
# ones as well. A basic guide on the most important (/unique) hyperparameters of
# each clustering algorithm can be found here:
# https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
# =============================================================================


# ADD COMMAND TO CREAETE CLUSTERING METHOD HERE
model =


# =============================================================================




    
# Let's train our model.
=============================================================================


# ADD COMMAND TO TRAIN MODEL HERE 


# =============================================================================



# For this project, the 'Silhuette' metric is suitable for the model's evaluation. =============================================================================


# ADD COMMANDS TO COMPUTE METRIC HERE
print()

# =============================================================================