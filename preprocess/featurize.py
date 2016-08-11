from gensim.models import Doc2Vec

training_sentences = '../data/allComms.txt'
d2v_mod = Doc2Vec.load('../data/imdb_all.d2v')
outfile = '../data/training_vectors.txt'
printOut = open(outfile, 'w')


with open(training_sentences, 'r') as f:
    for line in f:
        splits = line.rstrip('\n').split(' ')
        ID = splits[0]
        relevance = splits[1]
        sentType = splits[2]
        sent = splits[3:]
        vec = map(str, d2v_mod.infer_vector(sent))
        if sentType == 'Q:':
            qid = ID
            q_vec = vec
            # it's a question
        elif sentType == 'P:':
            combinedVec = q_vec + vec
            vecString = ''
            for n, v in enumerate(combinedVec):
                vecString += ' ' + v
            resultStr = '{} {}{}\n'.format(relevance, qid, vecString)
            printOut.write(resultStr)
            # it's a passage
        else:
            raise ValueError("Must be either a question or answer")
    printOut.close()
