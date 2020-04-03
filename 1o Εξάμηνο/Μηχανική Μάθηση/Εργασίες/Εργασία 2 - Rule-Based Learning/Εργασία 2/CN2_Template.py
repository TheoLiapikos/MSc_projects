# =============================================================================
# HOMEWORK 2 - RULE-BASED LEARNING
# CN2 ALGORITHM TEMPLATE
# Complete the missing code by implementing the necessary commands.
# For ANY questions/problems/help, email me: arislaza@csd.auth.gr
# =============================================================================


# For this project, the only thing that we will need to import is the "Orange" library.
# However, before importing it, you must first install the library into Python.
# Read the instructions on how to do that (it might be a bit trickier than usual!)
# =============================================================================


# IMPORT LIBRARY HERE (trivial but necessary...)
import Orange

# =============================================================================



# Load 'wine' dataset
# =============================================================================


# ADD COMMAND TO LOAD TRAIN AND TEST DATA HERE
wineData = 

# =============================================================================




# Define the learner that will be trained with the data.
# Try two different learners: an '(Ordered) Learner' and an 'UnorderedLearner'.
# =============================================================================


# ADD COMMAND TO DEFINE LEARNER HERE
learner = 


# =============================================================================




# At this step we shall configure the parameters of our learner.
# We can set the evaluator/heuristic ('Entropy', 'Laplace' or 'WRAcc'),
# 'beam_width' (in the range of 3-10), 'min_covered_examples' (start from 7-8 and make your way up), 
# and 'max_rule_length' (usual values are in the range of 2-5).
# They are located deep inside the 'learner', within the 'rule_finder' class.
# =============================================================================


# ADD COMMANDS TO CONFIGURE THE LEARNER HERE


# =============================================================================



# We want to test our model now. The CrossValidation() function will do all the
# work in this case, which includes splitting the whole dataset into train and test subsets, 
# then train the model, and produce results.
# So, simply call the CrossValidation() function from the 'testing' library
# and use as input arguments 1) the dataset and 2) the learner.
# Note that the 'learner' argument should be in array form, i.e. '[learner]'.
results = 




# As for the required metrics, you can get them using the 'evaluation.scoring' library.
# The 'average' parameter of each metric is used while measuring scores to perform 
# a type of averaging on the data. DON'T WORRY MUCH ABOUT THAT JUST YET (AGAIN). USE EITHER 
# 'MICRO' OR 'MACRO' (preferably 'macro', at least for final results).
# =============================================================================


# # ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
print()
print()
print()


# =============================================================================



# Ok, now let's train our learner manually to see how it can classify our data
# using rules.You just want to feed it some data- nothing else.
# =============================================================================


# ADD COMMAND TO TRAIN THE LEARNER HERE
classifier = 


# =============================================================================




# Now we can print the derived rules. To do that, we need to iterate through 
# the 'rule_list' of our classifier.
for rule in ____:
    print()