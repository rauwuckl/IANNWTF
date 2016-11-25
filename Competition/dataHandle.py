import pickle
import numpy as np
import matplotlib.pyplot as plt

class DataHandle:
    def __init__(self):
        self.generalData = pickle.load(open("iannwtf-svhn/trainDataEqualized.pickle", 'rb'))
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


    """draws N random images and applys normalization acording to the normFunction
    then it plots the original and the normalized images side by side"""
    def applyAndShow(self, N, normFunction):
        if(N%2 != 0): 
            raise ValueException('passt nicht')
        indx = np.random.choice(len(self.generalLabel), N)
        img = self.generalData[indx]
        label = self.generalLabel[indx]

        plt.figure()
        for i in range(N):
            plt.subplot(N/2,4,i*2+1)
            plt.imshow(img[i].reshape(img[i].shape[0:2]), cmap='gray')
            plt.title("original #{}".format(label[i]))
            plt.subplot(N/2,4,i*2+2)
            norm = normFunction(img[i])
            #print("{}-{}".format(np.amin(norm), np.amax(norm)))

            plt.imshow(norm.reshape(norm.shape[0:2]), cmap='gray')
            plt.title("normalized #{}".format(label[i]))
        plt.show()
        

"""applys histogram equalization on an image. the output image will be of the same shape with all pixels in the range [-0.5, 0.5]"""
def histogramEqualization(img):
    originalShape = img.shape
    img = img.reshape(originalShape[0:2])+128
    hist, _ = np.histogram(img.flatten(), range(0, 256))
    normHist = hist.astype(float)/np.prod(np.shape(img))
        
    integral = np.cumsum(normHist)
    transferLookup = (np.insert(integral, 0, 0))

    equalized = np.zeros(img.shape)
    for i,value in enumerate(transferLookup):
        equalized[img==i] = value

    #if(np.amax(equalized)<0.5):
    #    print(np.amin(img))
    #    print(hist)
    #    print(transferLookup)

    return equalized.reshape(originalShape)-0.5

"""applys 'function' to all images in the pickle file 'pathOriginal' and saves the result in 'pathTarget'"""
def preprocessImages(pathOriginal, pathTarget, function):
    images = pickle.load(open(pathOriginal, 'rb'))
    result = np.empty(np.shape(images))
    for i,img in enumerate(images):
        result[i] = function(img)
        if i%100==0:
            print("already {} done".format(i/len(images)))

    pickle.dump(result, open(pathTarget, "wb")) 

    



