import pickle
import numpy as np

class DataHandle:
    def __init__(self):
        self.generalData = pickle.load(open("iannwtf-svhn/trainData.pickle", 'rb'))
        self.generalLabel = pickle.load(open("iannwtf-svhn/trainLabels.pickle", 'rb'))%10

        self.indxValidation = np.random.choice(len(self.generalLabel), len(self.generalLabel)//10, replace=False)#
        self.indxTraining = np.delete(range(len(self.generalLabel)), self.indxValidation)

    """get random batches of size N
    @return data, labels
    where data is an array of shape [N, 28, 28, 1]  containing 28x28 gray images
    and labels is of shape [N] containing ints in [0,10)
    """
    def getTrainingBatch(self, N):
        indx = np.random.choice(self.indxTraining,N, replace=True)
        return self.generalData[indx], self.generalLabel[indx]

    """see get TrainingBach"""
    def getValidationIterator(self, N):
        lower = 0
        upper = 0
        while lower<len(self.indxValidation):
            upper = min(lower+N,len(self.indxValidation))
            yield self.generalData[self.indxValidation[lower:upper]], self.generalLabel[self.indxValidation[lower:upper]]
            lower = upper
            

    def getValidationData(self):
        return self.generalData[self.indxValidation], self.generalLabel[self.indxValidation]

