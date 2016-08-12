from gensim.models import Doc2Vec

training_sentences = '../data/aquaint_label_train.txt'
d2v_mod = Doc2Vec.load('../data/aquaint.d2v')
outfile = '../data/training_vectors_aquaint.txt'
printOut = open(outfile, 'w')


with open(training_sentences, 'r') as f:
    for line in f:
        splits = line.rstrip('\n').split(' ')
        pid = splits[0]
        relevance = splits[1]
        sentType = splits[2]
        sent = splits[3:]
        vec = map(str, d2v_mod.infer_vector(sent))
        if sentType == 'Q:':
            qid = pid
            q_vec = vec
            # it's a question
        elif sentType == 'P:':
            combinedVec = q_vec + vec
            vecString = ' '.join(combinedVec)
            resultStr = '{}\t{}\t{}\t{}\n'.format(relevance, qid, pid, vecString)
            printOut.write(resultStr)
            # it's a passage
        else:
            raise ValueError("Must be either a question or answer")
    printOut.close()
