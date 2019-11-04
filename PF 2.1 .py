#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:43:00 2019

@author: dpbmac-05
"""

#downlaoding the text from january to october
import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)

# Sample URL
#
# http://content.guardianapis.com/search?from-date=2016-01-02&
# to-date=2016-01-02&order-by=newest&show-fields=all&page-size=200
# &api-key=your-api-key-goes-here

# Change this for your API key:
MY_API_KEY = 'dfa0463d-a096-4655-acea-5053cfbf6cdf'

API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "", # leave empty, change start_date / end_date variables instead
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY}

# Update these dates to suit your own needs.
start_date = date(2019, 6, 1)
end_date = date(2019,10, 30)

dayrange = range((end_date - start_date).days + 1)
for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')
    fname = join(ARTICLES_DIR, datestr + '.json')
    if not exists(fname):
        # then let's download it
        print("Downloading", datestr)
        all_results = []
        my_params['from-date'] = datestr
        my_params['to-date'] = datestr
        current_page = 1
        total_pages = 1
        while current_page <= total_pages:
            print("...page", current_page)
            my_params['page'] = current_page
            resp = requests.get(API_ENDPOINT, my_params)
            data = resp.json()
            all_results.extend(data['response']['results'])
            # if there is more than one page
            current_page += 1
            total_pages = data['response']['pages']

        with open(fname, 'w') as f:
            print("Writing to", fname)

            # re-serialize it for pretty indentation
            f.write(json.dumps(all_results, indent=2))

import json
import os
import pandas as pd


# Update to the directory that contains your json files
# Note the trailing /
directory_name = "theguardian/collection/"

ids = list()
texts = list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                ids.append(id)
                texts.append(text)
# I have chosen to download from June 1st 2019 to oktober 30th 2019


### NB!!! The dataset stopped working monday night??? Suddently the text value 
### is only a full stop, and when I try to run the script, is gives the error:
### ValueError: Sample larger than population or is negative   
### I tried to start from the beginning, even downloading a new api,
### but it gives the same result. Don't know how to solve this problem :(

####################### Looking at the data ##########################

# Printing number of texts = 32949
print("Number of texts: " + str(len(texts)))
#taking a look at some of the texts to get a feel for the material.
import random
random.sample(texts,2)

# number of all words before any parsing: 28.865.021
all_words = list()
for text in text:
  words = text.split()
  all_words.extend(words)
#prints number of words
print("Word count is: "+ str(len(all_words)))

#sample of words before parsing (this result seems fine)
import random
random.sample(all_words, 10)
#['Pearson,','enough','made','who','was,','Conservatives','an','giving',
# 'something','on']

#number of unique words = 688192
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count is: " + str(unique_word_count))

#average word length: 4.902622658753652 ()
words = all_words
average = sum(len(word) for word in words) / len(words)
print('Average word length is ' + str(average))


# tokenizing the string 'text' in sentenses in order to count them:
from nltk.tokenize import sent_tokenize
tokenized_sent_G = sent_tokenize(text)
import random 
random.sample(tokenized_sent_G, 10)

#number of sentences: 6392106
all_sentences = list()
for text in tokenized_sent_G:
  sentences = texts
  all_sentences.extend(sentences)
sentence_count = len(all_sentences)
print('Number of sentences: ' + str(sentence_count))  

#Calculating the average number of sentences pr article: 194
#in other words, these are pretty long articles
average_number_sent = (sentence_count) / len(texts)
print('Average text length is ' + (str(average_number_sent)))

######################### term document matrix ########################

from sklearn.feature_extraction.text import CountVectorizer
model_vect = CountVectorizer(stop_words = 'english', 
                             token_pattern = r'[a-zA-Z\-][a-zA-Z\-]{2,}')
texts_vect = model_vect.fit_transform(texts)
#returns number of rows = 32949, number of columns = 230422, number of cells = 9481715
print('Shape is : ' + str(texts_vect.shape))
print('Size is : ' + str(texts_vect.size))


#Term-document count matrix on the parameter "texts" with parsing for: 
#stopwords, token patterns
from sklearn.feature_extraction.text import CountVectorizer
model_vect = CountVectorizer(stop_words = 'english', 
                             token_pattern = r'[a-zA-Z\-][a-zA-Z\-]{2,}')
texts_vect = model_vect.fit_transform(texts)
#returns number of rows(articles) = 32949, number of columns = 230422, number of cells = 9481715
print('Shape is : ' + str(texts_vect.shape))
print('Size is : ' + str(texts_vect.size))



# Counting density =  0.9991264969790462
import numpy as np
dense_count = np.prod(texts_vect.shape)
zeros_count = dense_count - texts_vect.size
zeros_count / dense_count

#The indexes of the top-10 most used words.
counts = texts_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:10]
print('Top idxs: ' + str(top_idxs))

##The words belonging to the top-10 indexes.
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: " + str(top_words))

print(list(zip(top_words, top_idxs)))

#### Kopier
# 4.c. Show the document vectors for 10 random documents. Limit the vector to only the top-10 most used words.
# Note: To slice a sparse matrix use A[list1, :][:, list2] (see https://stackoverflow.com/questions/7609108/slicing-sparse-scipy-matrix).
import random
some_row_idxs = random.sample(range(0,len(texts)), 10)
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs))
sub_matrix = texts_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix

## 5. Apply TF-IDF weighting
# Tip: Use scikit's TfidfTransformer on previous document-term count matrix.
from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(texts_vect)
data_tfidf.shape

# 5.a. What are the top-10 terms based on TF-IDF? 
# Tip: Take the average (not sum) of TF-IDF scores per term.
freqs = data_tfidf.mean(axis=0).A1
top_idxs = (-freqs).argsort()[:10].tolist()
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print(list(zip(top_idxs, top_words)))


# 5.b. Show document-term weights matrix for 10 random texts and your top-10 terms.
# Bonus: Convert to pandas and add column and row names.
sub_matrix = data_tfidf[some_row_idxs, :][:,top_idxs].todense()
df = pd.DataFrame(columns=top_words, index=some_row_idxs, data=sub_matrix)
df


############### Making a Query #########################

term = 'Claude Juncker'
term

### I can't get this to work. I have tried feeding it different data, but 
# it keeps giving a long array, instead of a short list. 

#this bit gives a list of value "none". I can't figure out why.
term_idxs = [model_vect.vocabulary_.get(i) for i in text]
term_idxs

term_counts = [counts[idx] for idx in term_idxs]
term_counts

# creating a query.
query = (term)
query

#Transforming the query string into a query vector.
query_vect_counts = model_vect.transform([term])
query_vect = model_tfidf.transform(query_vect_counts)
query_vect


# The similarity between the query vector and each document.
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(query_vect, data_tfidf)
sims

#sims in sorted list
sims_sorted_idx = (-sims).argsort()
sims_sorted_idx

#the textual content of the best match.
texts[sims_sorted_idx[0,0]]

#################### Topic modeling ##################################

### I have tried to fit the query vect, but it gives a wierd result with #Out[20]: (1, 4)
### Did topic modeling for the full text instead, just to do something

#Fitting text to LDA 
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=4, random_state=0)
LDA.fit(texts_vect)

first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-10:]
for i in top_topic_words:
    print(model_vect.get_feature_names()[i])

#Displaying the top-10 terms and their weights for each topic. 
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([model_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = LDA.transform(texts_vect)
topic_values.shape

# Fitting data to LDA model
from sklearn.decomposition import LatentDirichletAllocation
model_lda = LatentDirichletAllocation(n_components=4, random_state=0)
data_lda = model_lda.fit_transform(texts_vect)
# Describe the shape of the resulting matrix.
import numpy as np
np.shape(data_lda)

#Showing the sults in a data frame
import pandas as pd
topic_names = ["Topic" + str(i) for i in range(LDA.n_components)]
doc_names = ["Doc" + str(i) for i in range(len(texts))]
df = pd.DataFrame(data=np.round(data_lda, 2), columns=topic_names, index=doc_names).head(10)
df

for i, term_weights in enumerate(model_lda.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = ["%s (%.3f)" % (model_vect.get_feature_names()[idx], 
                      term_weights[idx]) for idx in top_idxs]
    print("Topic %d: %s" % (i, ", ".join(top_words)))





