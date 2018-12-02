import pandas as pd
import csv

STOCK_PATH = "/Users/jenniferzou/Documents/MIT/Fall 2018/UROP/Stock List.xlsx"
STOCK_DESC_PATH = "/Users/jenniferzou/Documents/MIT/Fall 2018/UROP/Stock Descriptions.xlsx"
REUTERS_PATH = "/Users/jenniferzou/Documents/MIT/Fall 2018/UROP/reuters21578" #folder
GLOVE_6B_50D_PATH = "/Users/jenniferzou/Documents/MIT/Micronotes/Kaied/glove.6b.50d.txt"
GLOVE_6B_300D_PATH = "/Users/jenniferzou/Documents/MIT/Micronotes/Kaied/glove.6b.300d.txt"
encoding="utf-8"

#Quandl for closing price of 5 stocks

# TO-DO:
# download stock twitter data (use stream to iterate through list of stock tickers)
# download wikipedia data (use list of stock names)

stock_info = pd.read_excel(STOCK_PATH, header=0)
stock_desc = pd.read_excel(STOCK_DESC_PATH, header=0)
print(stock_info.head())

# access reuter's data
from nltk.corpus import reuters #ignore downloaded folder file path

all_docs = [reuters.raw(doc_id) for doc_id in reuters.fileids()]

# # access twitter api
# from twython import Twython  
# import json

# # Enter your keys/secrets as strings in the following fields
# credentials = {}  
# credentials['CONSUMER_KEY'] = 'iQHC5K8auC7cELxxlNIRawzQv'  
# credentials['CONSUMER_SECRET'] = 'EBY9C8v5iQZYzDxTPMfxicvJmdp0b5Cw8S5A5DM4gcoIUjaGYg' 
# credentials['ACCESS_TOKEN'] = '584280188-flkPxcXbwuHNTGVlxIrluylUc5tOoMlF44W8WgdA'  
# credentials['ACCESS_SECRET'] = 'ttU97C9Tb16EpXV7tOSW6l5LZhTiAyKE0x7FPitEYWJUK'

# # Instantiate an object
# python_tweets = Twython(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])

### EXAMPLE QUERY
# # Create our query
# query = {'q': 'learn python',  
#         'result_type': 'popular',
#         'count': 10,
#         'lang': 'en',
#         }
# # Search tweets
# dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}  
# for status in python_tweets.search(**query)['statuses']:  
#     dict_['user'].append(status['user']['screen_name'])
#     dict_['date'].append(status['created_at'])
#     dict_['text'].append(status['text'])
#     dict_['favorite_count'].append(status['favorite_count'])

# # Structure data in a pandas DataFrame for easier manipulation
# df = pd.DataFrame(dict_)  
# df.sort_values(by='favorite_count', inplace=True, ascending=False)  
# df.head(5)  

### EXAMPLE PROCESSING
# # Filter out unwanted data
# def process_tweet(tweet):  
#     d = {}
#     d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
#     d['text'] = tweet['text']
#     d['user'] = tweet['user']['screen_name']
#     d['user_loc'] = tweet['user']['location']
#     return d


# # Create a class that inherits TwythonStreamer
# class MyStreamer(TwythonStreamer):     

#     # Received data
#     def on_success(self, data):

#         # Only collect tweets in English
#         if data['lang'] == 'en':
#             tweet_data = process_tweet(data)
#             self.save_to_csv(tweet_data)

#     # Problem with the API
#     def on_error(self, status_code, data):
#         print(status_code, data)
#         self.disconnect()

#     # Save each tweet to csv file
#     def save_to_csv(self, tweet):
#         with open(r'saved_tweets.csv', 'a') as file:
#             writer = csv.writer(file)
#             writer.writerow(list(tweet.values()))

# # Instantiate from our streaming class
# stream = MyStreamer(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'],  
#                     credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])
# # Start the stream
# stream.statuses.filter(track='python') 

# access wikipedia api
# import wikipediaapi
import wikipedia

# wiki = wikipediaapi.Wikipedia('en')
def get_common(name):
	bad_words = [' corp',' co',' inc.',' inc',' ltd']
	name = name.lower() #.translate(str.maketrans('','',string.punctuation))
	for bad in bad_words:
		name = name.replace(bad,'')
	return name

# def get_page(name):
# 	print("Searching for %s" % get_common(name))
# 	results = wikipedia.search(name)
# 	# if len(results) > 0 and get_common(results[0])==get_common(name): #results[0].lower().translate(str.maketrans('','',string.punctuation)) == name.lower().translate(str.maketrans('','',string.punctuation)):
# 	# 	name = results[0]
# 	if len(results) > 0:
# 		name = results[0]
# 	page = wikipedia.page(name.replace(' ','_'))
# 	# name = name.lower().replace(' ','_')
# 	# page = wiki.page(name)
# 	if page.exists():
# 		return page.text
# 	return None

def get_page(name):
	print("Searching for %s" % name)
	results = wikipedia.search(name)
	if len(results) > 0:
		content = wikipedia.page(results[0]).content
		return content.replace(' ','')
	return None

wiki_pages = {}
for name in stock_info['Name']:
	page = get_page(name)
	if page is not None:
		wiki_pages[name] = page
	else:
		print("Missing page for %s" % name)

#thomson reuters eikon access

#train vectors on wikipedia and reuters data
import numpy as np
import string
from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.naive_bayes import BernoulliNB, MultinomialNB
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import StratifiedShuffleSplit

tokens = []
labels = []
for doc_id in reuters.fileids():
	tokens.append(word_tokenize(reuters.raw(doc_id).translate(str.maketrans('','',string.punctuation)).lower()))
	labels.append(doc_id)
for name, content in wiki_pages.items():
	tokens.append(word_tokenize(content.translate(str.maketrans('','',string.punctuation)).lower()))
	labels.append(name)

X, y = np.array(tokens), np.array(labels)
print ("total examples %s" % len(y))

import struct 

glove_small = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_6B_300D_PATH, "rb") as infile:
	for line in infile:
		parts = line.split()
		word = parts[0].decode(encoding)
		if (word in all_words):
			nums=np.array(parts[1:], dtype=np.float32)
			glove_small[word] = nums

# train word2vec on all the texts - both training and test set
# we're not using test labels, just texts so this is fine
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2) #default: skip-gram, negative sampling = 5, no hierarchical softmax
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
model_1 = Word2Vec(X, size=100, window=5, min_count=5, workers=2, negative=2) #1: skip-gram, negative sampling = 2, no hierarchical softmax
w2v_1 = {w: vec for w, vec in zip(model_1.wv.index2word, model_1.wv.syn0)}
model_2 = Word2Vec(X, size=100, window=5, min_count=5, workers=2, sg=0) #2: cbow, negative sampling = 5, no hierarchical softmax
w2v_2 = {w: vec for w, vec in zip(model_2.wv.index2word, model_2.wv.syn0)}
model_3 = Word2Vec(X, size=100, window=5, min_count=5, workers=2, sg=0, negative=2) #3: cbow, negative sampling = 5, no hierarchical softmax
w2v_3 = {w: vec for w, vec in zip(model_3.wv.index2word, model_3.wv.syn0)}
model_4 = Word2Vec(X, size=100, window=5, min_count=5, workers=2, hs=1) #4: skip-gram, negative sampling = 5, hierarchical softmax
w2v_4 = {w: vec for w, vec in zip(model_4.wv.index2word, model_4.wv.syn0)}
model_5 = Word2Vec(X, size=100, window=5, min_count=5, workers=2, hs=1, negative=2) #5: skip-gram, negative sampling = 2, hierarchical softmax
w2v_5 = {w: vec for w, vec in zip(model_5.wv.index2word, model_5.wv.syn0)}

# # start with the classics - naive bayes of the multinomial and bernoulli varieties
# # with either pure counts or tfidf features
# mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
# bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
# bern_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
# # SVM - which is supposed to be more or less state of the art 
# # http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
# svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
# svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

class MeanEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		if len(word2vec)>0:
			self.dim=len(word2vec[next(iter(glove_small))])
		else:
			self.dim=0
			
	def fit(self, X, y):
		return self 

	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
					or [np.zeros(self.dim)], axis=0)
			for words in X
		])

	
# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		self.word2weight = None
		if len(word2vec)>0:
			self.dim=len(word2vec[next(iter(glove_small))])
		else:
			self.dim=0
		
	def fit(self, X, y):
		tfidf = TfidfVectorizer(analyzer=lambda x: x)
		tfidf.fit(X)
		# if a word was never seen - it must be at least as infrequent
		# as any of the known words - so the default idf is the max of 
		# known idf's
		max_idf = max(tfidf.idf_)
		self.word2weight = defaultdict(
			lambda: max_idf, 
			[(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
	
		return self
	
	def transform(self, X):
		return np.array([
				np.mean([self.word2vec[w] * self.word2weight[w]
						 for w in words if w in self.word2vec] or
						[np.zeros(self.dim)], axis=0)
				for words in X
			])

# vectorizers
vect1 = MeanEmbeddingVectorizer(glove_small) #baseline GloVe
vect2 = TfidfEmbeddingVectorizer(glove_small) #baseline GloVe
vect3 = MeanEmbeddingVectorizer(w2v)
vect4 = TfidfEmbeddingVectorizer(w2v)
vect3 = MeanEmbeddingVectorizer(w2v_1)
vect4 = TfidfEmbeddingVectorizer(w2v_1)
vect3 = MeanEmbeddingVectorizer(w2v_2)
vect4 = TfidfEmbeddingVectorizer(w2v_2)
vect3 = MeanEmbeddingVectorizer(w2v_3)
vect4 = TfidfEmbeddingVectorizer(w2v_3)
vect3 = MeanEmbeddingVectorizer(w2v_4)
vect4 = TfidfEmbeddingVectorizer(w2v_4)
vect3 = MeanEmbeddingVectorizer(w2v_5)
vect4 = TfidfEmbeddingVectorizer(w2v_5)



