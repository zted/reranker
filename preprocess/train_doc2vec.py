# gensim modules
from random import shuffle

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


sources = {'/home/ted/COE/reranker/data/aquaint_train_unsup.txt': 'AQUAINT',
           # '/home/ted/Downloads/imdbstuff/test-neg.txt': 'TEST_NEG',
           # '/home/ted/Downloads/imdbstuff/test-pos.txt': 'TEST_POS',
           # '/home/ted/Downloads/imdbstuff/train-pos.txt': 'TRAIN_POS',
           # '/home/ted/Downloads/imdbstuff/train-neg.txt': 'TRAIN_NEG',
           # '/home/ted/Downloads/imdbstuff/train-unsup.txt': 'TRAIN_UNSUP',
           }

sentences = LabeledLineSentence(sources)
model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)

model.build_vocab(sentences.to_array())
for epoch in range(10):
    model.train(sentences.sentences_perm())
    print('Finished {} epochs'.format(epoch))

model.save('../data/aquaint.d2v')
