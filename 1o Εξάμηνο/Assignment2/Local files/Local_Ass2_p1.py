#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import bokeh
import gensim

import numpy as np
import os
from random import shuffle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Import necessary Libraries
import urllib.request
import zipfile
# import lxml.etree

# Download the dataset - XXXMB
#urllib.request.urlretrieve("https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip", filename="wikitext-103-v1.zip")


# Extract only the data of interest
# From the .zip file open and read only the training tokens
with zipfile.ZipFile('wikitext-103-v1.zip', 'r') as z:
  doc = z.open('wikitext-103/wiki.train.tokens', 'r').read()

# The first 500 bytes of data
print(doc[:500])

# Convert bytes to string and then split to paragraphs
doc_str = doc.decode("utf-8")
doc_para  = doc_str.split('\n')

print(doc_para[:2])

#For every paragraph
doc_para_noEmpties = []
count = 0
for para in doc_para:
    count += 1
    para = re.sub(r'\s+', ' ',para)
    if para != ' ':
        para = para.lower()
        para = re.sub(r'[^a-z0-9]+', ' ',para)
        para = re.sub(r'\s+', ' ',para)
        para = para.split(' ')
        para = [word for word in para if word not in stopwords.words('english')]
        para = para[1:-1]
        doc_para_noEmpties.append(para)
    if(count%5000 == 0):
        print(count)

###############################################################################
# Τοπική αποθήκευση των προκατεργασμένων δεδομένων σε αρχείο .csv για να αποφεύγω
# την παραπάνω χρονοβόρα διαδικασία
import csv

with open("preproc_no_stop_words_data.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(doc_para_noEmpties)

# Ανάκτηση των τοπικών δεδομένων από το αρχείο .csv
doc_para_noEmpties = []
with open("preproc_no_stop_words_data.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        doc_para_noEmpties.append(row)

###############################################################################

print(doc_para_noEmpties[:6])
print(type(doc_para_noEmpties))


from gensim.models import Word2Vec

model = Word2Vec(window=4,size=100,sg=1,min_count=10,workers = -1)
model.build_vocab(doc_para_noEmpties)  # prepare the model vocabulary
model.train(doc_para_noEmpties,total_examples=model.corpus_count,epochs=model.iter)

# Vocabulary size
vocab = model.wv.vocab
vocab1 = model.wv.index2word
print
print(vocab1[:10])
print(model.wv.index2word[:10])

model.wv['valkyria']

model.wv.most_similar('valkyria')

# 10 most frequent words
mfw = model.wv.index2word[:10]
similar_pairs = {}
for p1 in mfw:
    pairs = model.wv.most_similar(p1, restrict_vocab=10)
    for p2,sim in pairs:
        similar_pairs[(p1, p2)] = sim

import operator

sorted_similar_pairs = sorted(similar_pairs.items(), key=operator.itemgetter(1), reverse=True)[0:10:2]


# A function taking as input a trained word2vec model and two strings and manually
# computes and returns their cosine distance
def cosVecDist(w2vModel, str1, str2):
    # Use model to find vector representation of strings
    vstr1 = w2vModel.wv[str1]
    vstr2 = w2vModel.wv[str2]
    # Cosine distance of two vectors equals to the dot product of the vectors
    # divided by the product of vectrors' legths
    # The dot product of two 1-D vectors can be computed using numpy.dot() function
    dotProd = np.dot(vstr1,vstr2)
    # The length of a 1-D vector can be computed using numpy.linalg.norm() function
    # The default value of ord parameter (ord=None), returns the 2-norm of vectors
    length1 = np.linalg.norm(vstr1, ord=None)
    length2 = np.linalg.norm(vstr2, ord=None)
    # the cosine distance of the two vectors
    cosDist = dotProd/(length1*length2)
    return(cosDist)

# Compare the two approaches
cosDist1 = cosVecDist(model, 'first', 'unk')
cosDist2 = model.wv.similarity('first', 'unk')



# 20 most frequent words in 4 models
from gensim.models import KeyedVectors

# WikiText
wiki20mfw = model.wv.index2word[:20]

model40 = KeyedVectors.load_word2vec_format('40/model.txt', binary=False)
model40_20mfw = model40.index2word[:20]

model75 = KeyedVectors.load_word2vec_format('75/model.txt', binary=False)
model75_20mfw = model75.index2word[:20]

model82 = KeyedVectors.load_word2vec_format('82/model.txt', binary=False)
model82_20mfw = model82.index2word[:20]

file = open('40/model.txt', 'r')
print(file.read(5000))







