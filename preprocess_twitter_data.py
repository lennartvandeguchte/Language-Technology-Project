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
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


# for i in range(len(your_list)):
#     your_list[i][3] = preprocess(your_list[i][3])


##-------to remove stop words------------
punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

count_all = Counter()
terms_all = []
for i in range(len(your_list)):
    # Create a list with all the terms
    #terms_all = [term for term in preprocess(your_list[i][3])]
    # Delete stop words in tweets
    your_list[i][3] = [term for term in preprocess(your_list[i][3]) if term not in stop]
    
    # Update the counter
    count_all.update(your_list[i][3])

    # Create bigrams
    
    print(i)
# Print the first 10 most frequent words
print(count_all.most_common(10))
# terms_bigram = bigrams(your_list[1][3])
# print(your_list[:][3])



with open('text_emotion_preprocessed.csv', 'w', newline='') as f:
   wr = csv.writer(f)
   wr.writerows(your_list)
