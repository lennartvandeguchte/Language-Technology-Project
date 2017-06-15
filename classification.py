import numpy as np
import csv
import gensim
import keras
import random

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models.word2vec import Word2Vec
LabeledSentence = gensim.models.doc2vec.LabeledSentence
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn import preprocessing 


############## Parameter Settings ##########################################
preprocessdata = True
happysadData = True
loadW2Vmodel = False
shuffleWordOrder = False
classifier = 'LSTM' #set to: MLP or LSTM

percentageTrain = 0.70
dropoutPar = 0.2
batch_size  = 100

#Word2Vec
epochsW2V = 200
n_dim = 400
min_count = 5
batch_sizeWord2vec  = 50
windowsize = 5

#MLP
epochsMLP = 100
numberHiddenUnitsMLP = 100


#LSTM
epochsLSTM = 20
numberHiddenUnitsLSTM = 100

#############################################################################

# Preprocesses data if 'preprocesdata' is set to True
if(preprocessdata):
    import preprocess_twitter_data

with open('text_emotion_preprocessed.csv', 'r') as f:
     reader = csv.reader(f)
     preprocessed_sentences = list(reader)



## ------- Set happysadData on True if you only want to test on happy and sad tweets----------------------
if happysadData:
    happiness_sadness = []
    count = 0
    for i in range(len(preprocessed_sentences)):
        if (preprocessed_sentences[i][1] == 'sadness'):
            count = count + 1
        if (preprocessed_sentences[i][1] == 'happiness') | (preprocessed_sentences[i][1] == 'sadness'):
            happiness_sadness.append(preprocessed_sentences[i])
    preprocessed_sentences = happiness_sadness 

# Shuffle Data
preprocessed_sentences = random.sample(preprocessed_sentences , len(preprocessed_sentences))

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



# Transform the list of strings to a list of lists so word2vec gets the correct input. Additionally some more preprocessing is done 
# such that only proper English words remain in the tweets. 
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


trainSentences = transformToListOfLists(trainSentences)
testSentences = transformToListOfLists(testSentences)


with open('finalPreprocessedSentences.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(trainSentences)

# Shuffles the word order within tweets if 'shuffleWordOrder' is set to True
if shuffleWordOrder:
    for i in range(len(trainSentences)):
        trainSentences[i] = random.sample(trainSentences[i] , len(trainSentences[i]))
    for j in range(len(testSentences)):
        testSentences[j] = random.sample(testSentences[j] , len(testSentences[j]))


# Transform categorical labels to integers
le = preprocessing.LabelEncoder()
le.fit(trainLabels)
trainLabels = le.transform(trainLabels)
le.fit(testLabels)
testLabels = le.transform(testLabels)


# Transform integer labels to one-hot label representations, only needed when using more than 2 classes
if happysadData == False: 
    trainLabels = np.array(trainLabels)
    oneHotTrainLabels =  np.identity(len(set(trainLabels)))[trainLabels-1]

    trainLabels = np.array(testLabels)
    oneHotTestLabels =  np.identity(len(set(trainLabels)))[testLabels-1]

# Build vocabulary and train word2vec. If loadW2Vmodel is set to true it loads the model from the file 'word2vecModel'
if loadW2Vmodel:
    tweet_w2v = Word2Vec.load('word2vecModel')
else:
    tweet_w2v = Word2Vec(size=n_dim, min_count=min_count, batch_words=batch_sizeWord2vec)
    tweet_w2v.build_vocab(trainSentences+testSentences)
    tweet_w2v.train(trainSentences+testSentences, total_examples=(len(trainSentences)+len(testSentences)), epochs=epochsW2V) 
    tweet_w2v.save('word2vecModel')

# #Uncomment to print the vocabulary
# for word in tweet_w2v.wv.vocab:
#     print(word)

## Basic 3-layered multilayer perceptron

if(classifier == 'MLP'):

    # Builds a tf-idf matrix
    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=min_count)
    matrix = vectorizer.fit_transform([x for x in trainSentences])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    print(matrix.shape)

    # Merge word embeddings of all words in a sentences, representing a bag-of-words model.
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


    # MLP setup by using Keras
    model = Sequential()
    model.add(Dense(numberHiddenUnitsMLP, activation='sigmoid', input_dim=n_dim, use_bias=True))
    model.add(Dropout(dropoutPar))
    if happysadData:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense((len(set(trainLabels))), activation='softmax'))
        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


    print('Train classifier...')
    if happysadData:
        history = model.fit(train_vecs_w2v, trainLabels, validation_data=(test_vecs_w2v, testLabels), epochs=epochsMLP, batch_size=batch_size, verbose=2)
        score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
    else: 
        history = model.fit(train_vecs_w2v, oneHotTrainLabels, validation_data=(test_vecs_w2v, oneHotTestLabels), epochs=epochsMLP, batch_size=batch_size, verbose=2)
        score = model.evaluate(test_vecs_w2v, oneHotTestLabels, batch_size=batch_size, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



# Long short-term memory (LSTM)
if(classifier == 'LSTM'):

    train_vecs_w2v =  [[] for _ in range(len(trainSentences))]
    test_vecs_w2v = [[] for _ in range(len(testSentences))]


    # Create word embeddings by using the word2vec model
    print(trainSentences[0])   
    for i in range(len(trainSentences)):
        for word in trainSentences[i]:
            try:
                train_vecs_w2v[i].append(tweet_w2v[word]) 
            except KeyError:
                continue
            

    for i in range(len(testSentences)):
        for word in testSentences[i]:
            try:
                test_vecs_w2v[i].append(tweet_w2v[word]) 
            except KeyError:
                continue

    # Let all input sequences have equal length by using padding. 
    train_vecs_w2v = sequence.pad_sequences(train_vecs_w2v, maxlen=len(max(train_vecs_w2v,key=len)), dtype='int32', padding='pre', truncating='pre', value=0.)
    test_vecs_w2v = sequence.pad_sequences(test_vecs_w2v, maxlen=len(max(train_vecs_w2v,key=len)), dtype='int32',padding='pre', truncating='pre', value=0.)

    # LSTM setupt by using Keras
    model = Sequential()
    model.add(LSTM(numberHiddenUnitsLSTM, input_shape = ( len(max(train_vecs_w2v,key=len)), n_dim), dropout=dropoutPar, activation='sigmoid'))
    if happysadData:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.add(Dense((len(set(trainLabels))), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    if happysadData:
        history = model.fit(train_vecs_w2v, trainLabels, validation_data=(test_vecs_w2v, testLabels), epochs=epochsLSTM, batch_size=batch_size)
        score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
    else: 
        history = model.fit(train_vecs_w2v, oneHotTrainLabels, validation_data=(test_vecs_w2v, oneHotTestLabels), epochs=epochsLSTM, batch_size=batch_size)
        score = model.evaluate(test_vecs_w2v, oneHotTestLabels, batch_size=batch_size, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





# Plot train and test accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig('plot')

# Plot train and test loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.savefig('plot')