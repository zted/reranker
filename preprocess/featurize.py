from gensim.models import Doc2Vec

training_sentences = '../data/allComms.txt'
d2v_mod = Doc2Vec.load('../data/imdb.d2v')
outfile = '../data/training_vectors.txt'
printOut = open(outfile, 'w')
# model.save_word2vec_format('./imdb.w2vformat')


with open(training_sentences, 'r') as f:
    for line in f:
        splits = line.rstrip('\n').split(' ')
        qid = splits[0]
        relevance = splits[1]
        sentType = splits[2]
        sent = splits[3:]
        vec = map(str, d2v_mod.infer_vector(sent))
        if sentType == 'Q:':
            q_vec = vec
            # it's a question
        elif sentType == 'P:':
            resultVec = q_vec + vec
            resultStr = '{} id:{} {}\n'.format(relevance, qid, ' '.join(resultVec))
            printOut.write(resultStr)
            # it's a passage
        else:
            raise ValueError("Must be either a question or answer")
    printOut.close()
