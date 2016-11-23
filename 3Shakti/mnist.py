import struct
import numpy as np
nValidationData = 10000

class MNIST():
    def __init__(self, directory = "./"):
        testIMG = self._load(directory + "t10k-images-idx3-ubyte")
        self.testData = np.reshape(self.normalizeData(testIMG), [-1, 28, 28, 1])
        self.testLabels = self._load(directory + "t10k-labels-idx1-ubyte", True)
        self.nTest = len(self.testLabels)

        generalData = self._load(directory + "train-images-idx3-ubyte")
        generalLabel = self._load(directory + "train-labels-idx1-ubyte", True)
        indxValidation = np.random.choice(len(generalLabel), nValidationData, replace=False)

        validationIMG = generalData[indxValidation]
        self.validationData = np.reshape(self.normalizeData(validationIMG), [-1, 28, 28, 1])
        self.validationLabels = generalLabel[indxValidation]
        self.nValidation = nValidationData

        trainingIMG = np.delete(generalData, indxValidation, axis=0)
        self.trainingData = np.reshape(self.normalizeData(trainingIMG), [-1, 28, 28, 1])
        self.trainingLabels = np.delete(generalLabel, indxValidation, axis=0)
        self.nTrain = len(self.trainingLabels)
    
    def normalizeData(self, data):
        #rows are instances
        ndata = (data)/255 - 0.5
        print("shape(data): {}, shape(ndata): {}".format(np.shape(data), np.shape(ndata)))
        return ndata
        
    
    def _load(self, path, labels = False):
        
        with open(path, "rb") as fd:
            magic, numberOfItems = struct.unpack(">ii", fd.read(8))
            if (not labels and magic != 2051) or (labels and magic != 2049):
                raise LookupError("Not a MNIST file")
            
            if not labels:
                rows, cols = struct.unpack(">II", fd.read(8))
                images = np.fromfile(fd, dtype = 'uint8')
                images = images.reshape((numberOfItems, rows, cols))
                return images
            else:
                labels = np.fromfile(fd, dtype = 'uint8')
                return labels

    def trainingBatch(self, N):
        indexes = np.random.choice(self.nTrain, N, replace=True)
        return self.trainingData[indexes], self.trainingLabels[indexes]

    def getOneHotVectoren(self, label):
        oneHotMatrix = np.zeros([len(label), 10])
        oneHotMatrix[range(len(label)), label]=1
        return oneHotMatrix

    def oneHotTrainBatch(self, N):
        indexes = np.random.choice(self.nTrain, N, replace=True)
        return self.trainingData[indexes], self.getOneHotVectoren(self.trainingLabels[indexes])
        

                


        
                





#m = MNIST()
