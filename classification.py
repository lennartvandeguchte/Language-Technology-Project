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
import matplotlib.pyplot as plt

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


############## Parameter Settings ##########################################
preprocessdata = False
happysadData = True
loadW2Vmodel = True
classifier = 'LSTM' #set to: 2-layer-network or LSTM

epochsW2V = 200
percentageTrain = 0.8
n_dim = 50
min_count = 3
batch_size  = 5
dropoutPar = 0.2

epochs2Layernet = 200
numberHiddenUnits2Layer = 100

epochsLSTM = 3
numberHiddenUnitsLSTM = 50

#############################################################################

if(preprocessdata):
    import preprocess_twitter_data

with open('text_emotion_preprocessed.csv', 'r') as f:
     reader = csv.reader(f)
     preprocessed_sentences = list(reader)


## ------- Set happysadData on True if you only want to test on happy and sad tweets----------------------
if happysadData:
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

with open('finalPreprocessedSentences.csv', 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerows(trainSentences)


le = preprocessing.LabelEncoder()
le.fit(trainLabels)
trainLabels = le.transform(trainLabels)

le.fit(testLabels)
testLabels = le.transform(testLabels)

# Build vocabulary and train word2vec
if loadW2Vmodel:
    tweet_w2v = Word2Vec.load('word2vecModel')
else:
    tweet_w2v = Word2Vec(size=n_dim, min_count=min_count, batch_words=1000)
    tweet_w2v.build_vocab(trainSentences+testSentences)
    tweet_w2v.train(trainSentences+testSentences, total_examples=10000, epochs=epochsW2V) #[x.words for x in tqdm(x_train)]
    tweet_w2v.save('word2vecModel')




print(trainSentences[1])

print(tweet_w2v[trainSentences[1]])

if(classifier == '2-layer-network'):

    print('building tf-idf matrix ...')
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=min_count)
    matrix = vectorizer.fit_transform([x for x in trainSentences])
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    print('vocab size :', len(tfidf))
    print(matrix.shape)



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




    ## Basic 2 layer neural network
    model = Sequential()
    model.add(Dense(numberHiddenUnits2Layer, activation='relu', input_dim=n_dim, use_bias=True))
    model.add(Dropout(dropoutPar))
    if(happysadData):
        model.add(Dense(1, activation='relu'))
    else:
        model.add(Dense(14, activation='softmax'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    print('Train classifier...')
    history = model.fit(train_vecs_w2v, trainLabels, validation_split=0.33, epochs=epochs2Layernet, batch_size=batch_size, verbose=2)


    score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    prediction = model.predict(test_vecs_w2v, batch_size=batch_size)
    print('First prediction:', prediction[0])


if(classifier == 'LSTM'):

    train_vecs_w2v = []
    train_vecs_w2v = []

    for i in range(len(trainSentences)):
        try:
            train_vecs_w2v[i] = tweet_w2v[trainSentences[i]]
        except KeyError:
            continue
    
    for i in range(len(testSentences)):
        try:
            test_vecs_w2v[i] = tweet_w2v[testSentences[i]]
        except KeyError:
            continue


    train_vecs_w2v = sequence.pad_sequences(train_vecs_w2v, maxlen=200, dtype='int32', padding='pre', truncating='pre', value=0.)
    test_vecs_w2v = sequence.pad_sequences(test_vecs_w2v, maxlen=200, dtype='int32',padding='pre', truncating='pre', value=0.)

#     print(len(train_vecs_w2v))
#     print(len(train_vecs_w2v[1]))
   
#     model = Sequential()
#     model.add(Embedding(output_dim=128, input_dim=30, input_length=200))
#     model.add(LSTM(numberHiddenUnitsLSTM, dropout=dropoutPar, activation='sigmoid'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#     print(model.summary())
#     history = model.fit(train_vecs_w2v, trainLabels, validation_data=(test_vecs_w2v, testLabels), epochs=epochsLSTM, batch_size=64)

#     score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
#     print('Test loss:', score[0])
#     print('Test accuracy:', score[1])

    
#     prediction = model.predict(test_vecs_w2v, batch_size=batch_size)
#     print('First prediction:', prediction[0])



#     # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()