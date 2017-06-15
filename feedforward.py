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

    print('joooo')
    print(train_vecs_w2v.shape)


    model = Sequential()
    model.add(Dense(numberHiddenUnits2Layer, activation='relu', input_dim=n_dim, use_bias=True))
    model.add(Dropout(dropoutPar))
    if happysadData:
        model.add(Dense(1, activation='relu'))
        model.compile(optimizer='rmspop', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.add(Dense(14, activation='softmax'))
        model.compile(optimizer='rmspop', loss='categorical_crossentropy', metrics=['accuracy'])


    print('Train classifier...')
    history = model.fit(train_vecs_w2v, trainLabels, validation_data=(test_vecs_w2v, testLabels), epochs=epochs2Layernet, batch_size=batch_size, verbose=2)


    score = model.evaluate(test_vecs_w2v, testLabels, batch_size=batch_size, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    prediction = model.predict(test_vecs_w2v, batch_size=batch_size)
    print('First prediction:', prediction[0])