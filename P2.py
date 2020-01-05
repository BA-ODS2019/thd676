#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:27:00 2020

@author: dpbmac-05
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 20:43:00 2019

@author: dpbmac-05
"""

#downlaoding the text from june to october
import json
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta

# This creates two subdirectories called "theguardian" and "collection"
ARTICLES_DIR = join('theguardian', 'collection')
makedirs(ARTICLES_DIR, exist_ok=True)


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
### MINDRE DATASÆT
#end_date = date(2019,6,4)


# Downloading all data file
print("Downloadnig articles...")

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
directory_name = "theguardian/collection_full/"

print("\n-------------------------------------")
print("read all aricle texts......." )



ids = list()
strtexts = ""
texts = list()
allfields = list()
allheadlines = list()
allSections =list()
for filename in os.listdir(directory_name):
    if filename.endswith(".json"):
        with open(directory_name + filename) as json_file:
            data = json.load(json_file)
            for article in data:
                id = article['id']
                fields = article['fields']
                text = fields['bodyText'] if fields['bodyText'] else ""
                headline = article['webTitle'] #the variable headline is defined as the html-title, 'webTitle', in the corpus.
                sections = article['sectionName']# etc. 
                ids.append(id)
                texts.append(text)
                allSections.append(sections) 
                allfields.append(fields)
                allheadlines.append(headline)
            
# I have chosen to download from June 1st 2019 to oktober 30th 2019

#importing functions
import random
import nltk 


####################### Looking at the data ##########################


# Printing number of texts: 32954
print("Number of texts: " + str(len(texts)))

# Taking a look at some of the texts to get a feel for the material.
import random
random.sample(texts,1)


# Number of all words before any parsing after word tokenizing
from nltk.tokenize import word_tokenize
all_words = word_tokenize(str(texts)) 
print("Total word count before any parsing: ", len(all_words))

# Cheking the result
random.sample(all_words,10)
#['Pearson,','enough','made','who','was,','Conservatives','an','giving',
# 'something','on']

# Number of unique words
unique_words = set(all_words)
unique_word_count = len(unique_words)
print("Unique word count is: " + str(unique_word_count))

# Average word length
words = all_words
average = sum(len(word) for word in words) / len(all_words)
print('Average word length is ' + str(average))

# Tokenizing sentenses in order to count them
from nltk.tokenize import sent_tokenize
all_sentences = sent_tokenize(str(texts))
print('Number of sentences: ', len(all_sentences))  

# Checking the result
random.sample(all_sentences, 10)

# Calculating the average number of sentences pr article 
average_number_sent = (len(all_sentences)/len(texts))
print('Average text length is ' , (average_number_sent))

# Calculating the average number of words pr article 
average_number_words = (len(all_words)/len(texts))
print('Average text length is ' , (average_number_words))




######################### term document matrix ########################

# Importing stopwords from nltk
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
     
# Appyling it to the countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
model_vect = CountVectorizer( token_pattern = r'[a-zA-Z\-][a-zA-Z\-]{2,}',
                             stop_words = 'english')
texts_vect = model_vect.fit_transform(texts)
print(model_vect.vocabulary_)

# Number of cells(occorences) 
print('Shape is : ' + str(texts_vect.shape))
print('Size is : ' + str(texts_vect.size))


# Counting density 
import numpy as np
dense_count = np.prod(texts_vect.shape)
zeros_count = dense_count - texts_vect.size
zeros_count / dense_count

#The indexes of the top-10 most used words.
counts = texts_vect.sum(axis=0).A1
top_idxs = (-counts).argsort()[:10]
print('Top idxs: ' + str(top_idxs))


## The words belonging to the top-10 indexes.
inverted_vocabulary = dict([(idx, word) for word, idx in model_vect.vocabulary_.items()])
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print("Top words: " + str(top_words))

print(list(zip(top_words, top_idxs)))


# Showing the document vectors for 10 random documents
import random
some_row_idxs = random.sample(range(0,len(texts)), 10)
print("Selection: (%s x %s)" % (some_row_idxs, top_idxs))
sub_matrix = texts_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix

###################3 TF-IDF weighting ###################

# Applying TF-IDF weighting
from sklearn.feature_extraction.text import TfidfTransformer
model_tfidf = TfidfTransformer()
data_tfidf = model_tfidf.fit_transform(texts_vect)
data_tfidf.shape


# The top-10 terms based on TF-IDF
freqs = data_tfidf.mean(axis=0).A1
top_idxs = (-freqs).argsort()[:10].tolist()
top_words = [inverted_vocabulary[idx] for idx in top_idxs]
print(list(zip(top_idxs, top_words)))


# Showing document-term weights matrix for 10 random texts and the top-10 terms.
sub_matrix = data_tfidf[some_row_idxs, :][:,top_idxs].todense()
df = pd.DataFrame(columns=top_words, index=some_row_idxs, data=sub_matrix)
df

sub_matrix = texts_vect[some_row_idxs, :][:, top_idxs].todense()
sub_matrix

############### Making a Query #########################
# Selecting terms
terms = ['Trump', 'tweets']
terms

term_idxs = [model_vect.vocabulary_.get(word) for word in terms]
term_counts = [counts[idx] for idx in term_idxs]
term_counts

# Perform a search
# Use as query: 'try to buy car insurance'
query = " ".join(terms)
query

query_vect_counts = model_vect.transform([query])
query_vect = model_tfidf.transform(query_vect_counts)
query_vect

# The similarity between the query vector and each document.
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(query_vect, data_tfidf)
sims

# Sims in sorted list
sims_sorted_idx = (-sims).argsort()
sims_sorted_idx

# The textual content of the best match.
texts[sims_sorted_idx[0,0]]

# Dataframe showing the 10 best results
print("Shape of 2-D array sims: (%i, %i)" % (len(sims), len(sims[0,:])) )
df = pd.DataFrame(data=zip(sims_sorted_idx[0,:], sims[0,sims_sorted_idx[0,:]]), columns=["index", "cosine_similarity"])
df[0:10]

#################### Topic modeling ##################################

# Fitting data to LDA 
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=4, random_state=0)
LDA.fit(texts_vect)

first_topic = LDA.components_[0]
top_topic_words = first_topic.argsort()[-2:]
for i in top_topic_words:
    print(model_vect.get_feature_names()[i])

# Displaying the top-10 terms and their weights for each topic. 
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([model_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = LDA.transform(query_vect)
topic_values.shape

# Showing results in a wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for i, term_weights in enumerate(LDA.components_):
    top_idxs = (-term_weights).argsort()[:10]
    top_words = [model_vect.get_feature_names()[idx] for idx in top_idxs]
    word_freqs = dict(zip(top_words, term_weights[top_idxs]))
    wc = WordCloud(background_color="white",width=300,height=300, max_words=10).generate_from_frequencies(word_freqs)
    plt.subplot(2, 2, i+1)
    plt.imshow(wc)










