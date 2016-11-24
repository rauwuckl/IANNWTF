import pickle
import numpy as np

class DataHandle:
    def __init__(self):
        self.generalData = pickle.load(open("iannwtf-svhn/trainData.pickle", 'rb'))
        self.generalLabel = pickle.load(open("iannwtf-svhn/trainLabels.pickle", 'rb'))%10

        self.indxValidation = np.random.choice(len(self.generalLabel), len(self.generalLabel)//8, replace=False)#
        self.indxTraining = np.delete(range(len(self.generalLabel)), self.indxValidation)

    def getTrainingBatch(self, N):
        indx = np.random.choice(self.indxTraining,N, replace=True)
        return self.generalData[indx], self.generalLabel[indx]

    def getValidationData(self):
        return self.generalData[self.indxValidation], self.generalLabel[self.indxValidation]

