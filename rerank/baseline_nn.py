import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import numpy


def batch(x_data, y_data, vocab_dim, embedding_weights, static, n=64, shuffle=False):
    """
    batchify training examples, so that not all of them need to be loaded into
    memory at once
    :param x_data: all x data in indices
    :param y_data: tags
    :param vocab_dim: vocabulary dimension
    :param embedding_weights: mapping from word indices to word embeddings
    :param static: determines whether or not embeddings are the output
    :param n: batch size
    :param shuffle: whether or not to shuffle them
    :return: word embeddings corresponding to x indices, and corresponding tags
    """
    l = len(x_data)
    # shuffle the data
    if shuffle:
        randIndices = np.random.permutation(l)
        x_data = np.array([x_data[i] for i in randIndices])
        y_data = np.array([y_data[i] for i in randIndices])

    for ndx in range(0, l, n):
        x_data_subset = x_data[ndx:min(ndx + n, l)]
        y_data_subset = y_data[ndx:min(ndx + n, l)]
        if static:
            x_out = np.zeros([len(x_data_subset), x_data.shape[1], vocab_dim])
            for i, example in enumerate(x_data_subset):
                for j, word in enumerate(example):
                    x_out[i][j] = embedding_weights[word]
            x_data_subset = x_out
        yield x_data_subset, y_data_subset


def train_nn(dataset, epochs, batch_size=5):
    # fix random seed for reproducibility
    Y, X, IDs = load_data(dataset)
    input_dim = X.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    X_train = X[1000:]
    Y_train = Y[1000:]
    X_val = X[0:1000]
    Y_val = Y[0:1000]

    tmpweights = '../model/weights.hdf5'
    checkpointer = ModelCheckpoint(filepath=tmpweights, monitor='val_acc', verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
              validation_data=(X_val, Y_val), callbacks=[checkpointer])


def load_data(someFile):
    y = []
    x = []
    ID = []
    with open(someFile, 'r') as f:
        for line in f:
            splits = line.split('\t')
            relevance = int(splits[0])
            qid = splits[1]
            pid = splits[2]
            x_data = np.array(splits[3].rstrip('\n').split(' '), dtype=np.float32)
            y.append(relevance)
            x.append(x_data)
            ID.append(pid)
    return np.array(y, dtype=np.int8), np.array(x), ID


if __name__ == "__main__":

    TRAINFILE = '../data/training_vectors_aquaint.txt'
    EPOCHS = 20

    train_nn(TRAINFILE, EPOCHS)