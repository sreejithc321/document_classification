'''
Categorizes each review as positive or negative.
Dataset - NLTK Movie Reviews Corpus
'''

import nltk
import random
from nltk.corpus import movie_reviews

import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

LIMIT = 1000

def create_dataset():
    '''Create dataset from movie reviews dataset'''
    documents = [(list(movie_reviews.words(fileid)), category)
                for category in movie_reviews.categories()
                for fileid in movie_reviews.fileids(category)]
    random.shuffle(documents)
    return documents

def contains_features(document): 
    '''Check given word is present or not in the document'''
    document_words = set(document) 
    word_features = list(all_words)[:LIMIT]
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features    

def bigram_features(words):
    '''Return bigrams in the document'''
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 200)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 

# Dataset
documents = create_dataset()
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# Model using single words
featuresets = [(contains_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)  
print 'Accuracy on Test Data - words: ',(nltk.classify.accuracy(classifier, test_set))

# Model using bigrams
featuresets = [(bigram_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = nltk.NaiveBayesClassifier.train(train_set)  
print 'Accuracy on Test Data - bigrams: ',(nltk.classify.accuracy(classifier, test_set))

# Sentiment Analysis
review = raw_input('Enter a reiew : ')
result = classifier.classify(bigram_features(review))
if result=='neg':
    print '\n Negative review'
else:
    print '\n Positive review'    