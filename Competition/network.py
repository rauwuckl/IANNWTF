import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import struct
import dataHandle

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

my_activation_function = tf.nn.relu

learningRate = 1e-4
gdsOptimizer = tf.train.AdamOptimizer(learningRate)

# images
x = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Here we input the correct answers
desired = tf.placeholder(tf.int64, [None])

# Reshape x to a 4d tensor, with the second and third dimensions corresponding
# to image width and height, and the final dimension corresponding to the number
# of color channels.
# x_image = tf.reshape(x, [-1,28,28,1])

# Note about weight(shape)
#   First three entries represent the kernel size: 5x5x1, where 1 is channel gray-scale
#   4th entry are the number of feature maps
# Feed conv kernels to input x
# Stride one. Zero padding --> the output is the same size as the input.
conv_1 = fun_conv(x, fun_weight([5,5,1,32]))

# Calculate neuron outputs
act_1 = my_activation_function(conv_1 + fun_bias([32]))

# Apply max pooling to outputs
pool_1 = fun_pool(act_1)
    
# Apply 64 kernels of size 5x5x64
# Third paramenter is 32 because from pool_1 we get 32
# feature maps of size 14x14 [14,14,32]
# 32 color channels? (1 color channel)*(32 feature maps)???
conv_2 = fun_conv(pool_1, fun_weight([5,5,32,64]))

# Calculate neuron outputs
act_2 = my_activation_function(conv_2 + fun_bias([64]))

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
output_ron_1 = my_activation_function(tf.matmul(pool_2_flat, w_ron_1) + b_ron_1)

# Do the same for last layer
w_ron_2 = fun_weight([1024,10])
b_ron_2 = fun_bias([10])

# no activation function yet (see next Training)
output_ron_2 = tf.matmul(output_ron_1, w_ron_2) + b_ron_2

crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(output_ron_2, desired)
crossEntropy = tf.reduce_mean(crossEntropy)

trainingStep = gdsOptimizer.minimize(crossEntropy)


prediction = tf.argmax(output_ron_2, 1)
nCorrect = tf.equal(prediction, desired)
#accuracy = tf.equal(tf.argmax(output_ron_2,1), tf.argmax(desired,1))
accuracy = tf.reduce_mean(tf.cast(nCorrect, tf.float32))

miniBatchSize = 300
plotStepSize = 25


#accuracyFigure, accuracyAxis = plt.subplots(1,1)
#weightFigure, weightAxes = plt.subplots(2,5)
data = dataHandle.DataHandle()

savePath = 'saves/run1.cpkt'
restorePath = 'saves/run1.cpkt'
#restorePath = None

trainingAccuracy = []
validationAccuracy = []
netPredictions = []
def trainNetwork(trainingSteps):
    with tf.Session() as session:
        saver = tf.train.Saver()
        if(restorePath is None):
            session.run(tf.initialize_all_variables())
            print("initialized variables")
        else:
            saver.restore(session, restorePath)
            print("restored variables")
        
        for step in range(trainingSteps):
            images, labels = data.getTrainingBatch(miniBatchSize)
            print(labels)
            acc, netPrediction, _ = session.run([accuracy, prediction, trainingStep], feed_dict= {x: images, desired: labels})

            trainingAccuracy.append(acc)
            print("{} accuracy at step {}".format(acc, step))
            
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

        if(not savePath is None):
            path= saver.save(session, savePath)
            print("path: {}".format(path))

        plt.ion()
        plt.figure()
        plt.scatter(range(trainingSteps), netPrediction, c='r', label='Networks Estimate')
        plt.scatter(range(trainingSteps), netPrediction, c='r', label='Networks Estimate')

        _accuracy = [session.run(accuracy, feed_dict = {x: images, desired: labels}) for images, labels in data.getValidationIterator(300)]
        print("accuracys: {}".format(_accuracy))
        # generator version is A LOT faster then putting evertything into one tensor (or I am not geting all)
        _accuracy = np.mean(_accuracy)
        print("Test accuracy: ", _accuracy)
    
