import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
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


def predict_ranking(evalFile, outFile):

    X, qids, pids = load_data(evalFile)
    input_dim = X[0].shape[1]

    assert len(pids[0]) == len(X[0])

    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    weightsFile = '../model/weights.hdf5'
    model.load_weights(weightsFile)

    Y_p = []
    for x in X:
        Y_p.append(model.predict(x))

    f = open(outFile, 'w')

    for n, qid in enumerate(qids):
        tupes = zip(Y_p[n], pids[n])
        sortedTupes = sorted(tupes, key=lambda x: x[0], reverse=True)
        for n, (y, pid) in enumerate(sortedTupes):
            f.write('{}\tITER\t{}\t{}\t{}\tSOMEID\n'.format(qid, pid, n, 1001-n))


def load_data(someFile):
    x = []
    QIDS = []
    PIDS = []
    previous = ''
    with open(someFile, 'r') as f:
        for line in f:
            splits = line.split('\t')
            qid = splits[1]
            pid = splits[2]
            if qid != previous:
                PIDS.append([])
                previous = qid
                QIDS.append(qid)
                x.append([])
            x_data = np.array(splits[3].rstrip('\n').split(' '), dtype=np.float32)
            PIDS[-1].append(pid)
            x[-1].append(x_data)
    x = map(np.array, x)
    return x, QIDS, PIDS


if __name__ == "__main__":

    EVAL = '../data/dev_vectors_good.txt'
    RESULTSFILE = '../data/predictions_dev_good.txt'
    predict_ranking(EVAL, RESULTSFILE)