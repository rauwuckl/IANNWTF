import numpy as np
import matplotlib.pyplot as plt
import mnist
import tensorflow as tf

# plot with plt.imshow(m.trainingData[532], cmap='gray')
# plt.imshow(a[53].reshape([28,28]), cmap='gray') #a normalized and flattend


class MLP():
    def __init__(self):
        self.mnistData = mnist.MNIST()
        self.buildNetwork()

    def buildNetwork(self):
        # Input
        self.inputData = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        # self.inputData = tf.reshape(inpData, [-1, 28, 28, 1])
        # reshapedInpData = tf.reshape(self.inputData , [-1, 28, 28, 1])
        self.targets = tf.placeholder(tf.int32, shape=[None])

        # Layer 1
        ## Conv
        self.kernelL1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
        self.biasL1   = tf.Variable(tf.constant(0.1, shape=[32]))
        self.featuresL1 = tf.nn.conv2d(self.inputData, self.kernelL1, strides=[1,1,1,1], padding="SAME")
        self.activationL1 = tf.nn.relu(self.featuresL1 + self.biasL1)
        ## maxpool
        self.outL1 = tf.nn.max_pool(self.activationL1, ksize=[1,2,2,1], strides=[1,2,2,1] ,padding="SAME")
        print(self.outL1.get_shape())
        # (batch, height, width, features)

        #Layer 2
        ## Conv
        self.kernelL2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
        self.biasL2   = tf.Variable(tf.constant(0.1, shape=[64]))
        self.featuresL2 = tf.nn.conv2d(self.outL1, self.kernelL2, strides=[1,1,1,1], padding="SAME")
        self.activationL2 = tf.nn.relu(self.featuresL2 + self.biasL2)
        ## maxpool
        self.outL2 = tf.nn.max_pool(self.activationL2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
        print(self.outL2.get_shape())

        #Fully 1
        self.convOutput = tf.reshape(self.outL2, [-1, 7*7*64])
        # the first one for the batches. should stay the same
        self.weightsF1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
        self.biasF1 = tf.Variable(tf.constant(0.1 , shape=[1024]))
        self.activationF1 = tf.nn.relu(tf.matmul(self.convOutput, self.weightsF1) + self.biasF1)

        #Fully 2 
        self.weightsF2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
        self.biasF2 = tf.Variable(tf.constant(0.1, shape=[10]))
        self.output = tf.matmul(self.activationF1, self.weightsF2) + self.biasF2

        

        print(self.output.get_shape())
        # Minimizer:
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.output, self.targets))

        print("crossentropie: {} ".format(self.cross_entropy.get_shape()))
        self.minimizer = tf.train.AdamOptimizer(1e-04).minimize(self.cross_entropy)

        # For accuracy:
        correct_prediction = tf.equal(tf.cast(tf.argmax(self.output,1), tf.int32), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.gradients = tf.gradients(self.cross_entropy, [self.kernelL1, self.kernelL2, self.weightsF1, self.weightsF2])
        self.gradients = [tf.sqrt(tf.reduce_mean(tf.square(grad))) for grad in self.gradients]

        

    def batchGD(self, sess, Nepochs, batchSize=300):
        Nbatches = self.mnistData.nTrain // batchSize
        trainAccuracys = []
        valiAccuracys = []
        gradients = []
        for j in range(Nepochs):
            accuracysInEpoch = []
            for i in range(Nbatches):
                data, label = self.mnistData.trainingBatch(batchSize)
                _, acc, grad= sess.run([self.minimizer, self.accuracy, self.gradients], feed_dict={self.inputData: data, self.targets:label})
                #print(acc)
                #accuracysInEpoch.append(acc)
                trainAccuracys.append(acc)
                gradients.append(grad)
            #trainAccuracys.append(np.mean(accuracysInEpoch))
            #valiAccuracys.append(self.validatePerformance(sess))
            #print("training: {}, test: {}".format(trainAccuracys[-1], valiAccuracys[-1]))
            
        return trainAccuracys, valiAccuracys, np.array(gradients)
            

    def validatePerformance(self, sess):
        data = self.mnistData.validationData
        label = self.mnistData.validationLabels
        perf = sess.run(self.accuracy, feed_dict={self.inputData: data, self.targets: label})
        return perf
        
        


m = MLP()
N = 1

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print('All Variables initizlized')
    #print("performance before: {}".format(m.validatePerformance(sess)))
    trainA, valiA, grad = m.batchGD(sess, N)
    print("performance after: {}".format(m.validatePerformance(sess)))

print(np.shape(trainA))
print(grad)
plt.figure()
plt.plot(trainA, label="training accuracy")
plt.plot(valiA, label="validation accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(grad[:,0], label='Layer 0')
plt.plot(grad[:,1], label='Layer 1')
plt.plot(grad[:,2], label='Layer 2')
plt.plot(grad[:,3], label='Layer 3')
plt.legend()

plt.show()

#with tf.Session() as sess:
#    data = m.mnistData.validationData
#    label = m.mnistData.validationLabels
#    sess.run(tf.initialize_all_variables())
#
#    perf = sess.run(m.accuracy, feed_dict={m.inputData: data, m.targets: label})
#    print("peformacne before: {}".format(perf))
#
#    for i in range(N):
#        data, label = m.mnistData.trainingBatch(100)#oneHotTrainBatch(100)
#        _, w = sess.run([m.minimizer, m.weights], feed_dict={m.inputData: data, m.targets:label})
#        if(i%1000==0):
#            print(w)
#        
#
#            
#    #Validate:
#    perf = sess.run(m.accuracy, feed_dict={m.inputData: data, m.targets: label})
#    print("peformacne after: {}".format(perf))
                    
    
        

    

        
        
    
        
