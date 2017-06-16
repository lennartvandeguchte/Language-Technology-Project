import csv
import re
from collections import Counter
import operator 
from nltk.corpus import stopwords
import string
from nltk import bigrams 
from nltk import ngrams

with open('text_emotion.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth 
        [=)] # Happy Smiles
        [:)]
        [:D]
        [=(] # Sad smiles
        [:(]
        [;(]
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
 # Remove usernames, urls and punctuations. 
def preprocess(s, lowercase=False):
    s = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",s).split())
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


##-------to remove stop words------------
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

count_all = Counter()
terms_all = []
for i in range(len(your_list)): 
    # Delete stop words in tweets
    your_list[i][3] = [term for term in preprocess(your_list[i][3]) if term not in stop]
    # Update the counter
    count_all.update(your_list[i][3])
    print('Preprocessed tweets:', i)



with open('text_emotion_preprocessed.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(your_list[1:])
