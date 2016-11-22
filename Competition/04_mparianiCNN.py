import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import struct

class MNIST():
    def __init__(self, directory = r"C:\Users\mpariani\Documents\UnivOsnabrueck\Third_Semester\ANNs_TensorFlow"):
        self.testData = self._load(directory + "t10k-images.idx3-ubyte")
        self.testLabels = self._load(directory + "t10k-labels.idx1-ubyte", True)
        self.trainingData = self._load(directory + "train-images.idx3-ubyte")
        self.trainingLabels = self._load(directory + "train-labels.idx1-ubyte", True)
        
        randomIndices = np.random.choice(len(self.trainingLabels), 6000, replace = False)
        self.validationData = self.trainingData[randomIndices]
        self.validationLabels = self.trainingLabels[randomIndices]
        self.trainingData = np.delete(self.trainingData, randomIndices, axis = 0)
        self.trainingLabels = np.delete(self.trainingLabels, randomIndices)
    
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
    
    def getRandomSample(self, size):
        randomIndices = np.random.choice(len(self.trainingLabels), size, replace = False)
        return self.trainingData[randomIndices], self.trainingLabels[randomIndices]
    
    def getValidationData(self):
        return self.validationData, self.validationLabels
    
    def getTestData(self):
        return self.testData, self.testLabels
        
mnist = MNIST("./")

# MNIST images
x = tf.placeholder(tf.float32, [None, 784])

# Here we input the correct answers
desired = tf.placeholder(tf.int64, [None])

def fun_weight(shape):
    initvar = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initvar)

def fun_bias(shape):
    initvar = tf.constant(0.1, shape = shape)
    return tf.Variable(initvar)

def fun_conv(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = "SAME")
    
def fun_pool(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")


# Reshape x to a 4d tensor, with the second and third dimensions corresponding
# to image width and height, and the final dimension corresponding to the number
# of color channels.
x_image = tf.reshape(x, [-1,28,28,1])

# Note about weight(shape)
#   First three entries represent the kernel size: 5x5x1, where 1 is channel gray-scale
#   4th entry are the number of feature maps
# Feed conv kernels to input x
# Stride one. Zero padding --> the output is the same size as the input.
conv_1 = fun_conv(x_image, fun_weight([5,5,1,32]))

# Calculate neuron outputs
act_1 = tf.nn.tanh(conv_1 + fun_bias([32]))

# Apply max pooling to outputs
pool_1 = fun_pool(act_1)
    
# Apply 64 kernels of size 5x5x64
# Third paramenter is 32 because from pool_1 we get 32
# feature maps of size 14x14 [14,14,32]
# 32 color channels? (1 color channel)*(32 feature maps)???
conv_2 = fun_conv(pool_1, fun_weight([5,5,32,64]))

# Calculate neuron outputs
act_2 = tf.nn.tanh(conv_2 + fun_bias([64]))

# Apply max pooling to outputs
pool_2 = fun_pool(act_2)

# 64 is the number of feature maps with size 7x7
# We project the output (7*7*64) on 1024 neurons
# Each neuron is connected and receives the whole weighted output of the previous layer.
w_ron_1 = fun_weight([7*7*64, 1024])
b_ron_1 = fun_bias([1024])

# print(pool_2.shape)

# Reshape for matrix mult --> why -1?
pool_2_flat = tf.reshape(pool_2, [-1, 7*7*64])

# Apply activation function
output_ron_1 = tf.nn.tanh(tf.matmul(pool_2_flat, w_ron_1) + b_ron_1)

# Do the same for last layer
w_ron_2 = fun_weight([1024,10])
b_ron_2 = fun_bias([10])

# no activation function yet (see next Training)
output_ron_2 = tf.matmul(output_ron_1, w_ron_2) + b_ron_2

crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_ron_2, desired)
crossEntropy = tf.reduce_mean(crossEntropy)

learningRate = 1e-4
gdsOptimizer = tf.train.AdamOptimizer(learningRate)
trainingStep = gdsOptimizer.minimize(crossEntropy)

accuracy = tf.equal(tf.argmax(tf.nn.softmax(output_ron_2), 1), desired)
#accuracy = tf.equal(tf.argmax(output_ron_2,1), tf.argmax(desired,1))
accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))

miniBatchSize = 300
plotStepSize = 25

trainingAccuracy = np.ones(trainingSteps)
validationAccuracy = np.ones(trainingSteps)

#accuracyFigure, accuracyAxis = plt.subplots(1,1)
#weightFigure, weightAxes = plt.subplots(2,5)

def trainNetwork(trainingSteps):
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        
        for step in range(trainingSteps):
            images, labels = mnist.getRandomSample(miniBatchSize)
            images = images.reshape([-1, 784])
            trainingAccuracy[step], _ = session.run([accuracy, trainingStep], feed_dict= {x: images, desired: labels})
            
            #if step % plotStepSize == 0 or step == trainingSteps - 1:
                #images, labels = mnist.getValidationData()
                #images = images.reshape([-1, 784])
                
                #_accuracy, _weights = session.run([accuracy, weights], feed_dict= {x: images, desired: labels})
                #if step != trainingSteps - 1:
                #    validationAccuracy[step:step+plotStepSize] = [_accuracy] * plotStepSize
                #accuracyAxis.cla()
                #accuracyAxis.plot(trainingAccuracy, color = 'b')
                #accuracyAxis.plot(validationAccuracy, color = 'r')
                #accuracyFigure.canvas.draw()

                #for i in range(10):
                #    weight = _weights[:,i].reshape(28, 28)
                #    weightAxes[i // 5, i % 5].cla()
                #    weightAxes[i // 5, i % 5].matshow(weight, cmap = plt.get_cmap('bwr'))
                #weightFigure.canvas.draw()

        images, labels = mnist.getTestData()
        images = images.reshape([-1, 784])
        _accuracy = session.run(accuracy, feed_dict = {x: images, desired: labels})
        print("Test accuracy: ", _accuracy)
    
