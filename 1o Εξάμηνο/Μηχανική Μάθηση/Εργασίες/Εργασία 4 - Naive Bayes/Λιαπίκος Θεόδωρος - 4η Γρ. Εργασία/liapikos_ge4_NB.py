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
from sklearn import datasets, model_selection, metrics, naive_bayes

# We also need to import 'make_pipeline' from the 'pipeline' module.
from sklearn.pipeline import make_pipeline

# We are working with text, so we need an appropriate package
# that shall vectorize words within our texts.
# 'TfidfVectorizer' from 'feature_extraction.text'.
from sklearn.feature_extraction.text import TfidfVectorizer

# 'matplotlib.pyplot' and 'seaborn' are ncessary as well,
# for plotting confusion matrix.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Το DataSet έχει ήδη διαχωρισμένα σύνολα training και test. Αφού όμως ο διαχωρισμός
# θα γίνει παρακω από εμάς, επιλέγω την παράμετρο subset='all' για να κατέβουν
# όλα τα δεδομένα
textData = datasets.fetch_20newsgroups(subset='all')

# Store features and target variable into 'X' and 'y'.
X = textData.data
y = textData.target


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
# 'random_state' parameter should be fixed to a constant integer (i.e. 0) so that the same split
# will occur every time this script is run.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 42)



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
NBalpha = 0.1 # This is the smoothing parameter for Laplace/Lidstone smoothing
max_f = None  # Max # of features used, (None, 1000, 5000, 12500)
ngram_r = (1,1)  # Size of n-grams extracted, (min,max): (1,1), (2,2), (3,3)

model = make_pipeline(TfidfVectorizer(max_features = max_f, ngram_range = ngram_r),
                      naive_bayes.MultinomialNB(alpha = NBalpha))

# Let's train our model.
# =============================================================================

# ADD COMMAND TO TRAIN MODEL HERE 
model.fit(x_train, y_train)

# Ok, now let's predict output for the second subset
# =============================================================================

# ADD COMMAND TO MAKE PREDICTION HERE
y_predicted = model.predict(x_test)


# =============================================================================

# ADD COMMANDS TO COMPUTE METRICS HERE (AND PRINT ON CONSOLE)
recall = metrics.recall_score(y_test,y_predicted,average='macro')
precision = metrics.precision_score(y_test,y_predicted,average='macro')
f1 = metrics.f1_score(y_test,y_predicted,average='macro')

print("Recall: %.3f" % recall)
print("Precision: %.3f" % precision)
print("F1: %.3f" % f1)

# In order to plot the 'confusion_matrix', first grab it from the 'metrics' module
# and then throw it within the 'heatmap' method from the 'seaborn' module.
# =============================================================================

# ADD COMMANDS TO PLOT CONFUSION MATRIX
labels = textData.target_names

confusionMatrix = metrics.confusion_matrix(y_test,y_predicted)

plt.figure(figsize=(10,8))
plt.title('Multinomial NB - Confusion matrix (a = %.2f) [Prec = %.5f, Rec = %.5f, F1 = %.5f] \n'
          % (NBalpha, precision, recall, f1), size=20)
sns.heatmap(confusionMatrix, cmap="Blues")
locs, _ = plt.xticks()  # Οι θέσεις των labels των αξόνων
locsx = locs-0.7
plt.xticks(locsx, labels, rotation = 60, size=12)
plt.yticks(locs, labels, rotation = 0, size=12)
plt.xlabel('True output', size=20)
plt.ylabel('Predicted output', size=20)

plt.show()


# =============================================================================
