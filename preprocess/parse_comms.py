from concrete.util import CommunicationReader
import os

commDir = '/home/ted/COE/data/mini_aquaint/'
allComms = sorted(os.listdir(commDir))
outfile = open('/home/ted/COE/data/allComms.txt', 'w')

for commName in allComms:
    cr = CommunicationReader(commDir+commName)
    (comm, filename) = cr.next()
    for num, sec in enumerate(comm.sectionList):
        assert len(sec.sentenceList) == 1
        sent = sec.sentenceList[0]
        newSent = []
        for tok in sent.tokenization.tokenList.tokenList:
            newSent.append(tok.text.lower())
        resultStr = ' '.join(newSent)
        if num == 0:
            printStr = '{} Q: {}\n'.format(comm.id, resultStr)
            pass
        else:
            label = 1 if sec.label == 'positive' else 0
            printStr = '{}.{} {} P: {}\n'.format(comm.id, num, label, resultStr)
            pass
        outfile.write(printStr)