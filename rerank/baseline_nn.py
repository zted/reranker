import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
    seed = 7
    numpy.random.seed(seed)
    Y, X, IDs = load_data(dataset)
    print(Y[0:10])

    input_dim = X.shape[1]
    model = Sequential()
    model.add(Dense(20, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X, Y, nb_epoch=epochs, batch_size=batch_size)

    Y_p = model.predict(X[0:10])
    outfile = '../data/predictions.txt'
    with open(outfile, 'w') as f:
        for y in Y_p:
            n = np.round(y)
            f.write('{}\n'.format(int(n[0])))


def load_data(someFile):
    y = []
    x = []
    ID = []
    with open(someFile, 'r') as f:
        for line in f:
            splits = line.rstrip('\n').split(' ')
            relevance = int(splits[0])
            sentenceID = splits[1]
            y.append(relevance)
            x.append(np.array(splits[2:], dtype=np.float32))
            ID.append(sentenceID)
    return np.array(y, dtype=np.int8), np.array(x), ID


if __name__ == "__main__":

    TRAINFILE = '../data/training_vectors.txt'
    EPOCHS = 10

    train_nn(TRAINFILE, EPOCHS)