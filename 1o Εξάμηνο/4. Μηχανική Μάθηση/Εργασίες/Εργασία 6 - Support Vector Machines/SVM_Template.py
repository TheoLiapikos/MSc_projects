# =============================================================================
# HOMEWORK 6 - Support Vector Machines
# SUPPORT VECTOR MACHINE ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================

# From sklearn, we will import:
# 'datasets', for loading data
# 'model_selection' package, which will help validate our results
# 'metrics' package, for measuring scores
# 'svm' package, for creating and using a Support Vector Machine classfier
# 'preprocessing' package, for rescaling ('normalizing') our data
from sklearn import 

# We also need 'pandas' library to manipulate our data.
import

# Import credit card dataset from file. 
# Pandas will read this file as a 'dataframe' object (using 'read_csv' command)
# so we will treat that object accordingly.
# (FYI, a dataframe object has more infromation than just values, but we'll
# stick just to values for this project).
myData = 


# From myData object, read included features and target variable.
# You can select a specific row/column of the dataframe object by using 'iloc',
# as if you were accessing an array (tip: you can access the last row or column
# with 'iloc' using the '-1' index).
# Make sure that, after selecting (/accessing) the desired rows and columns,
# you extract their 'values', which is what you want to store in variables 'X' and 'y'.
X =
y = 

# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = 




# Suport Vector Machines, like other classification algorithms, are sensitive to the magnitudes 
# of the features' values. Since we already split our dataset into 'train' and 'test' sets,
# we must rescale them separately (but with the same scaler).
# So, we rescale train data to the [0,1] range using a 'MinMaxScaler()' from the 'preprocessing' package,
# from which we shall then call 'fit_transform' with our data as input.
# Note that, test data should be transformed ONLY (not fit+transformed), since we require the exact
# same scaler used for the train data. 
# =============================================================================

minMaxScaler = 
x_train = 
x_test = 


# =============================================================================




# Now we are ready to create our SVM classifier. Scikit-learn has more than one type of SVM classifier,
# so, let us all agree on using the 'SVC' classifier.
# C: This parameter, also called 'penalty', is used to control the 'decisiveness' of the SVM.
#    In essence, it is used to guide the SVM when deciding between creating a smooth surface,
#    i.e. larger margins in the hyperplane and thus more misclassifications (low C)
#    or choosing smaller separation margins and thus lowering misclassification levels (high C).
#    Usually, we go for a high C value (i.e. more correct classifications).
# kernel: This is one of the most important parameters in a Support Vector Machine classifier.
#         Simply put, it provides a way for the classifier to transform the current representation
#         of the samples in the hyperplane, into another kind of representation (i.e.  create a mapping),
#         where it is easier to separate the data.
#         Available options for this parameters are 'linear', 'poly' (polynomial),
#         'rbf' (Radial Basis Function) and 'sigmoid'.
# degree: Only used when kernel = 'poly'. This parameter is used to define the
#         degree of the polynomial.
# gamma: Only used when working with 'rbf', 'rbf' and 'sigmoid' kernels. The effect 
#        of this parameter resembles the effect of the number of neighbors in 
#        K-Nearest Neighbors classification. A low 'gamma' value cannot produce great results
#        because it cannot 'see' the underlying shape that can 'hug' (or, group together) 
#        similar points in the hyperplane, while a high 'gamma' value is highly likely to
#        overfit the model (and thus will not be able to generalize well due to high variance and low bias).
#        Scikit-learn can choose a good value for gamma (passing 'auto' uses 1/n_features 
#        as a gamma value, although it will be replaced by something similar in the next version).
# Note that a good model can be produced by keeping a good balance between 'C' and 'gamma'!
# =============================================================================


model =


# =============================================================================




    
# Let's train our model.
# =============================================================================


# ADD COMMAND TO TRAIN MODEL HERE 


# =============================================================================




# Ok, now let's predict output for the second subset
# =============================================================================

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
print()
print()
print()

# =============================================================================