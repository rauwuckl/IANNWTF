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
        self.inputData = tf.placeholder(tf.float32, shape=[None, 784])
        self.targets = tf.placeholder(tf.int32, shape=[None])
        #self.targets = tf.placeholder(tf.float32, shape=[None, 10])

        self.weights = tf.Variable(tf.random_normal([784, 10], stddev=0.35))
        self.b = tf.Variable( tf.zeros([1, 10]))
        self.activationFirstLayer = tf.matmul(self.inputData, self.weights) + self.b #inputData*weights + bias

        self.outputFirstLayer = self.activationFirstLayer
        #best so far: sigmoid
        #self.prediction = tf.nn.softmax(self.activationFirstLayer)
        self.cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(self.outputFirstLayer, self.targets))
        print(self.cross_entropy.get_shape())

        #elementwiseEntropie = tf.reduce_sum(self.targets * tf.log(self.prediction), reduction_indices=[1])
        #self.cross_entropy = tf.reduce_mean(-elementwiseEntropie) 
        #print("cross entropie: {}, elementwiseEntropie: {}".format(self.cross_entropy.get_shape(), elementwiseEntropie.get_shape()))
        self.minimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)

        correct_prediction = tf.equal(tf.cast(tf.argmax(self.outputFirstLayer,1), tf.int32), self.targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        
        # act functions in tf.nn

    def batchGD(self, sess, Nepochs, batchSize=100):
        Nbatches = self.mnistData.nTrain // batchSize
        trainAccuracys = []
        valiAccuracys = []
        for j in range(Nepochs):
            accuracysInEpoch = []
            for i in range(Nbatches):
                data, label = self.mnistData.trainingBatch(100)
                _, acc= sess.run([self.minimizer, self.accuracy], feed_dict={self.inputData: data, self.targets:label})
                accuracysInEpoch.append(acc)
            trainAccuracys.append(np.mean(accuracysInEpoch))
            valiAccuracys.append(self.validatePerformance(sess))
            
        return trainAccuracys, valiAccuracys 
            

    def validatePerformance(self, sess):
        data = self.mnistData.validationData
        label = self.mnistData.validationLabels
        perf = sess.run(self.accuracy, feed_dict={self.inputData: data, self.targets: label})
        return perf
        
        


m = MLP()
N = 90

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print("performance before: {}".format(m.validatePerformance(sess)))
    trainA, valiA = m.batchGD(sess, N)
    print("performance after: {}".format(m.validatePerformance(sess)))

plt.figure()
plt.plot(trainA, label="training accuracy")
plt.plot(valiA, label="validation accuracy")
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
                    
    
        

    

        
        
    
        
