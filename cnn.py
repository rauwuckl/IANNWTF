   

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

epochs = 200
eta = 0.001
dens = 50
batchSize = 50
wavelength = 0.5
# Helper Methods
def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))
#####  CNN  #####
#x = tf.placeholder(tf.float32,shape=[None,28*28])
x = tf.placeholder(tf.float32,shape=[None,28,28])
label = tf.placeholder(tf.float32,shape=[None,10])

# First Conv Layer
kernels1 = weight_variable([5,5,1,32]) # 1 because we're dealing with greyscale images
biasC1 = bias_variable([32]) # Bias is the same for all neurons on a feature map
image = tf.reshape(x,[-1,28,28,1])
fmap1In = tf.nn.conv2d(image,kernels1,strides=[1,1,1,1],padding='SAME') + biasC1
fmap1Out = tf.nn.relu(fmap1In)
pool1 = tf.nn.max_pool(fmap1Out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# Second Conv Layer
kernels2 = weight_variable([5,5,32,64])
biasC2 = bias_variable([64])
fmap2in = tf.nn.conv2d(pool1,kernels2,strides=[1,1,1,1],padding='SAME') + biasC2
fmap2Out = tf.nn.relu(fmap2in)
pool2 = tf.nn.max_pool(fmap2Out,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
# First FCL
#pool2_flat = tf.reshape(pool2,[7*7*64])
pool2_flat = tf.reshape(pool2,[-1,7*7*64])
weights1 = weight_variable([7*7*64,1024])
biasF1 = bias_variable([1024])
f1in = tf.matmul(pool2_flat,weights1) + biasF1
f1out = tf.nn.relu(f1in)

# Second FCL
weights2 = weight_variable([1024,10])
biasF2 = bias_variable([10])
f2in = tf.matmul(f1out,weights2) + biasF2
OUT = tf.nn.softmax(f2in)

# --------------- BACKWARD STEP ----------
crEnt = tf.reduce_mean(-tf.reduce_sum(label*tf.log(OUT),reduction_indices=[1]))
backw = tf.train.AdamOptimizer(eta).minimize(crEnt)



val = np.zeros([int(epochs/dens)])
train = np.zeros([epochs])
ind = 0
with tf.Session() as sess: 
	sess.run(tf.initialize_all_variables())
	for k in range(epochs):
		print(k)
		trainImgs, trainLabs = mnist.train.next_batch(batchSize)
		trainImgs = np.reshape(trainImgs,[batchSize,28,28])
		#trainImgsFilt = sobelImageSet(trainImgs)
		sess.run(backw,feed_dict={x:trainImgs, label:trainLabs})
		correct = tf.equal(tf.argmax(OUT,1), tf.argmax(label,1))
		accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
		train[k] = sess.run(accuracy, feed_dict={x:trainImgs, label:trainLabs})
		if (k+1) % dens == 0 :
			valImgs = np.reshape(mnist.validation.images,[len(mnist.validation.images),28,28])
			#valImgsFilt = sobelImageSet(valImgs)
			val[ind] = sess.run(accuracy, feed_dict={x:valImgs, label:mnist.validation.labels})
			print('Train Acc ', train[k], ' Val Acc ', val[ind])
			ind += 1
f = plt.figure()
x1 = np.linspace(50,epochs,epochs-50)
x2 = np.linspace(50,epochs,int(epochs/dens))
plt.plot(x1,train[50:])
plt.plot(x2,val)
plt.legend(['Training Accuracy', 'Validation Accurracy'])
f.savefig('CNN_MNIST_Sobel.png')


    
    
        
    
    


