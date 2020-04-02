# Working with Pandas' DataFrames

'''
Notes kept from Youtube Channel:
    https://www.youtube.com/channel/UCYCcf7stRPHTWtu0r4ZoXbA/playlists

DataFrame is a Collection of Series, each column is a different Series
Series is a 'list' of objects of the SAME type
Datas in a DataFrame are either NUMERICAL (type: ints, floats...) or NON-NUMERICAL
(strings, recognized as type: object)
It's better ALWAYS use Names for columns' indexing

'''
import pandas as pd
import numpy as np

# Some interesting DataSets from Internet to work with
drinks = pd.read_csv('http://bit.ly/drinksbycountry')
movies = pd.read_csv('http://bit.ly/imdbratings')
train = pd.read_csv('http://bit.ly/kaggletrain')
ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])


# HOW TO CREATE A DATAFRAME
# 1. Inserting datas from a local file or URL
df = pd.read_csv('test.csv')
df = pd.read_csv('http://bit.ly/imdbratings')
# 2. Create an empty DF
df = pd.DataFrame()
# 3. Add Columns
df['ColName'] = ['assadas','dsas', 'dseee', 'sdad', 'opipi']  # Create empty column
df['ColName2'] = [1,2,3,4,5]  # Create column and insert values
#df['ColName3'] = [6,7]  # ERROR, all new columns should have same size as previous ones
df['ColName3'] = [6,7,8,9,10]  # Successfully creates new column with same size
# 4. Add a new ROW at the bottom
# a. Create an empty pandas Series object and append to existin DF. Will create
# a new EMPTY row
new_row = pd.Series()
df = df.append(new_row, ignore_index=1)
# b. Create an pandas Series object giving values as many as the existing columns
# and append to existin DF. Will create a new FILLED row
new_row = pd.Series([78789,'dfds','dfds'])
df = df.append(new_row, ignore_index=1)
# ΠΡΟΣΟΧΗ Το παραπάνω λειτουργεί σωστά μόνο αν οι στήλες του DF ΔΕΝ έχουν όνομα,
# αλλιώς πάει και δημιουργεί νέες στήλες για να εισάγει τα νέα δεδομένα
# Η ΛΥΣΗ είναι να δηλώνω στη νέα γραμμή τα ονόματα των στηλών που με ενδιαφέρουν
new_row = pd.Series([78789,'dfds','dfds'], index=['ColName','ColName2','ColName3'])
df = df.append(new_row, ignore_index=1)


# FROM NOW ON I'LL BE USING MOVIES DATASET
df = movies.copy()

# HOW TO DESCRIBE A DATAFRAME
df.info()  # Headers and type of data
df.describe()  # Statistical data of each numerical column (count,mean,sd,min,max etc)
df.describe(include=object)  # Statistical data of each categorical (object) column (count,freq etc)
df.shape  # The dimensions' sizes of the DF
df.size  # Total elements (cells) of DF including NaNs
df.head()  # The first 5 rows of DF
df.tail()  # The last 5 rows of DF


# HOW TO NAVIGATE TO A DATAFRAME
df[2]['title']  # Specific element, as accessing a 2d matrix
# .iloc will return ROWS and COLUMNS of the DF using the default numerical indexing
df.iloc[:]  # ALL rows and columns
df.iloc[3]  # Specific row
df.iloc[2:5]  # Specific rows
df.iloc[:,1:3]  # Specific columns
df.iloc[1:4,2:4] # Specific rows with specific columns
# .loc acts as .iloc but can also use categorical manual indexing (see below)
df.loc[3]  # Specific row
df.loc[1:7]  # Specific rows
df.loc[1:4,4:5]  # Column SLICING WON'T WORK
# Alternative access of single categorical column
df.title  # Full Series (column)
df.title[3:10]  # Slicing specific rows os Series


#### ΝΑ ΤΟ ΞΑΝΑΨΑΞΩ ΤΟ .loc ΔΕΝ ΤΟ ΠΟΛΥΚΑΤΑΛΑΒΑ
# HOW MANUALLY SET THE INDEX OF DF AND .loc USAGE
# You can specify a certain column to act as INDEX of the DF
df2 = df.set_index(df['title'])
df2.iloc[2]  # Still can use numerical index with .iloc
df2.iloc[2:7]
df2.loc['Pulp Fiction']  # or categorical indeces with .loc
df2.loc['Pulp Fiction':'Inception']  # Slicing rows' categorical indeces
# For the columns we can select categorical indeces
df2.loc[:]['title']  # Single column
df2.loc['Inception']['title':'duration']  # Slicing columns with single row
df2.loc['Inception':'Goodfellas']['title':'duration']  # NOT working with multiple rows
df2.loc[0:3]['title':'duration']  # NOT working
df2.loc[:]['title':'duration']  # still NOT working
# We can also use numerical values if they apear at the column indexing (names)
df2.loc[:][0]  # The 4th column with name 0
df2.loc[:][2]  # The 6th column with name 2
df2.loc[:][5]  # ERROR No column with name 5
df2.loc[:][0:2]  # RETURNS 0 AND 1 ROWS!!!!!
#### ΝΑ ΤΟ ΞΑΝΑΨΑΞΩ ΤΟ .loc ΔΕΝ ΤΟ ΠΟΛΥΚΑΤΑΛΑΒΑ


# RENAME column(s)
df2 = df.copy()
df2.rename(columns={'genre': 'genre2', 'duration': 'diarkeia'}, inplace=True)

# CONDITIONAL selection of rows. Setting conditions to Series
df[df.star_rating >= 8.5]  # Single condition
df[df['star_rating'] >= 8.5]  # Alternative way to select Series
df[(df.star_rating >= 8.5) & (df.genre == 'Action')]  # Σύζευξη of multiple conds
df[(df.star_rating >= 8.5) | (df.genre == 'Action')]  # Σύζευξη of multiple conds


# REPLACE values
df2 = df.copy()
df2['star_rating'][1] = 9.8  # Replace single mumerical value
df2.star_rating[1] = 9.97  # Alternative way to select Series

df2['genre'][1] = 'Porn'  # Replace single categorical value
df2.genre[1] = 'Porn'  # Alternative way
df2.content_rating[1:7] = 'NOT RATED'  # Slicing rows
# Mass replacing categorical values. In order to change the existing DF should
# use parameter inplace=True 
df2['genre'].replace('Action', 'Musical', inplace=True)
# ATTENTION If u just reassign the result to original DF, will end up with a sigle series
df2 = df2['genre'].replace('Action', 'Paparies')


# DROPPING columns and rows from DF
df2 = df.copy()
# Give the name of the column to delete and axis=1 (for columns)
df2.drop('content_rating', axis=1, inplace=True)
# To replace multiple columns give the list of their names
df2.drop(['title','content_rating'], axis=1, inplace=True)
# Dropping columns using numerical indexing
df2.drop(df2.columns[2:4], axis=1, inplace=True)
df2.drop(df2.columns[[2,5]], axis=1, inplace=True)  # Μη συνεχόμενες στήλες

# Dropping rows using numerical indexing
df2.drop(df2.index[1:3], inplace=True)
df2.drop(df2.index[[1,3]], inplace=True)
# Dropping rows based on conditions of Series
df2.drop(df2.index[df2.content_rating != 'R'], inplace=True)
df2 = df2[df2['content_rating'] != 'R']  # Alternative way of dropping rows not matching a condition


# FUNCTIONS u can apply to DF Series
# Functions on numerical Series
df.star_rating.max()
df.star_rating.min()
df.star_rating.mean()
df.star_rating.std()
df.star_rating.sum()
df.star_rating.count()
df.star_rating.unique()
df.star_rating.value_counts()  # the count of each category (Unique value) in Series
df.star_rating.cumsum()  # Κάνει συσσωρευτική άθροιση (όπως στις πιθανότητες)

# Functions on categorical Series
df.genre.max()  # ATTENTION returns the max LEXICOGRAPHICAL value
df.genre.min()  # ATTENTION returns the min LEXICOGRAPHICAL value
df.genre.count()
df.genre.unique()
df.genre.value_counts()  # the count of each category in Series

''' Description of SOME (not all) available functions
count 	Number of non-NA observations
sum 	Sum of values
mean 	Mean of values
mad 	Mean absolute deviation
median 	Arithmetic median of values
min 	Minimum
max 	Maximum
mode 	Mode
abs 	Absolute Value
prod 	Product of values
std 	Bessel-corrected sample standard deviation
var 	Unbiased variance
sem 	Standard error of the mean
skew 	Sample skewness (3rd moment)
kurt 	Sample kurtosis (4th moment)
quantile 	Sample quantile (value at %)
cumsum 	Cumulative sum
cumprod 	Cumulative product
cummax 	Cumulative maximum
cummin 	Cumulative minimum
unique  Unique values
'''


# NAN values
df2 = df.copy()
df2['genre'][1:10] = np.nan  # Inserting some NaN values
df2.info()  # Usefull to see which column has NaN values
# Usefull boolean functions to track NaN values
df.genre.isnull()
df.genre[df.genre.isnull() == True]
df.content_rating.notnull()


# GROUP BY a categorical column
# Returns the results of a FUNCION applied, grouped by the values of a categorical
# Series (column)
# Grouping by a single categorical column. Can Assing results to new DF
df.groupby('content_rating').mean()  # Function applied on ALL numerical Series
results = pd.Series(df.groupby('content_rating')['duration'].mean())  #  # Function applied on SPECIFIC numerical Series
# Grouping by multiple (list of) categorical columns
results = df.groupby(['content_rating', 'genre'])['star_rating'].count()

