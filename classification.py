import numpy as np
import csv
import gensim
import keras

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn import preprocessing 



# Uncomment to preprocess the twitter data
#import preprocess_twitter_data

############## Parameter Settings ##########################################
percentageTrain = 0.9
n_dim = 200
min_count = 10
batch_size  = 50

#############################################################################
with open('text_emotion_preprocessed.csv', 'r') as f:
     reader = csv.reader(f)
     preprocessed_sentences = list(reader)


## ------- Uncomment section if you only want to test on sad and happy tweets ----------------------
happiness_sadness = []
for i in range(len(preprocessed_sentences)):
    if (preprocessed_sentences[i][1] == 'happiness') | (preprocessed_sentences[i][1] == 'sadness'):
        happiness_sadness.append(preprocessed_sentences[i])
preprocessed_sentences = happiness_sadness 



trainSentences = []
trainLabels = []
testSentences = []
testLabels = []

for i in range(len(preprocessed_sentences)):
    if i<(percentageTrain*len(preprocessed_sentences)):
        trainSentences.append(preprocessed_sentences[i][3])
        trainLabels.append(preprocessed_sentences[i][1])
    else:
        testSentences.append(preprocessed_sentences[i][3])
        testLabels.append(preprocessed_sentences[i][1])


def transformToListOfLists(sentences):
    goodSentences = []  
    keep=set('qazwsxedcrfvtgbyhnujmikolp QAZWSXEDCRFVTGBYHNUJMIKOLP')
    for line in sentences:
        line = str(line)
        line = ''.join(filter(keep.__contains__, line))
        line = line.split()
        for word in line:
            if len(word)<3:
                line.remove(word)
        goodSentences.append(line)
    return goodSentences

# Transform the list of strings to a list of lists so word2vec gets the correct input
trainSentences = transformToListOfLists(trainSentences)
testSentences = transformToListOfLists(testSentences)

# Build vocabulary and train word2vec
tweet_w2v = Word2Vec(size=n_dim, min_count=min_count)
tweet_w2v.build_vocab(trainSentences)
tweet_w2v.train(trainSentences, total_examples=10000, epochs=3) #[x.words for x in tqdm(x_train)]


print('building tf-idf matrix ...')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x for x in trainSentences])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))



def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, trainSentences))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x, testSentences))])
test_vecs_w2v = scale(test_vecs_w2v)


# print(len(train_vecs_w2v[1]))
le = preprocessing.LabelEncoder()
le.fit(trainLabels)
trainLabels = le.transform(trainLabels)

le.fit(testLabels)
testLabels = le.transform(testLabels)

# testLabels = keras.utils.to_categorical(testLabels, num_classes=3)


## Basic 2 layer neural network
model = Sequential()
model.add(Dense(64, activation='sigmoid', input_dim=200))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('Train classifier...')
model.fit(train_vecs_w2v, trainLabels, epochs=200, batch_size=batch_size, verbose=2)


score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
print(score[0])
print(score[1])

preds = model.predict(test_vecs_w2v, batch_size=batch_size)




# max_review_length = 30
# trainSentences= sequence.pad_sequences(goodTrainSentences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.)
# testSentences = sequence.pad_sequences(testSentences, maxlen=None, dtype='int32',
#      padding='pre', truncating='pre', value=0.)

# create the model
# embedding_vecor_length = 32
# top_words = 5000
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(trainSentences, trainLabels, validation_data=(testSentences, testLabels), epochs=3, batch_size=64)


