
# coding: utf-8

# # Assignment 2 - MNIST
# 
# ## 3 Read the data
# The images are provided in a non-standardized binary format. You can download a python script from studip (03 mnist.py), which reads the data for you, or you can implement your own script, following the description of the file format on the MNIST database homepage.
# Make sure to modify the script such that you retrieve a training dataset, a validation dataset and the test dataset separately.

# In[84]:
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
epochs = 2000
# In[80]:

# Network
pic = tf.placeholder(tf.float32,[None,784],name='Picture')
weights = tf.Variable(tf.random_normal([784,10],mean=0,stddev=1),name='Weights')
bias = tf.Variable(tf.random_normal([1,10], mean=0,stddev=1),name='Bias')
inp = tf.add(tf.matmul(pic,weights),bias,name='Excitation')
out = tf.nn.softmax(inp,name='Activation')
labels = tf.placeholder(tf.float32,[None,10],name='Targets')

# Define Loss function and LR
crEnt = tf.reduce_mean(-tf.reduce_sum(labels*tf.log(out),reduction_indices = [1]),name='Loss') # reduce_mean for minibatches
backw = tf.train.GradientDescentOptimizer(0.15).minimize(crEnt)



# In[83]:

# Train Network
val = np.zeros([epochs])
trai = np.zeros([epochs])
init = tf.initialize_all_variables()
w = []
with tf.Session() as session:
    session.run(init)
    for e in range(epochs): # Train hundred times with batches of size 10
        trainImgs, trainLabs = mnist.train.next_batch(10)
        session.run(backw, feed_dict={pic: trainImgs, labels: trainLabs})
        correct = tf.equal(tf.argmax(labels,1),tf.argmax(out,1))
        perf = tf.reduce_mean(tf.cast(correct,tf.float32))
        val[e] = session.run(perf, feed_dict={pic: mnist.validation.images, labels: mnist.validation.labels})
        trai[e] = session.run(perf, feed_dict={pic:mnist.train.images, labels: mnist.train.labels})
        if e%25 == 0:
            print('Train Acc ',trai[e], 'Val Acc ',val[e])
            w.append(weights.eval().reshape([28,28,10]))

f = plt.figure()
plt.plot(trai)
plt.plot(val)
plt.legend(['Training Accuracy', 'Validation Accuracy'])
f.savefig('Accuracy.png')
p,s = plt.subplots(2,5)
s[0,0].imshow(w[0][:,:,0], cmap='seismic')
s[0,1].imshow(w[0][:,:,1],cmap='seismic')
s[0,2].imshow(w[0][:,:,2],cmap='seismic')
s[0,3].imshow(w[0][:,:,3],cmap='seismic')
s[0,4].imshow(w[0][:,:,4],cmap='seismic')
s[1,0].imshow(w[0][:,:,5],cmap='seismic')
s[1,1].imshow(w[0][:,:,6],cmap='seismic')
s[1,2].imshow(w[0][:,:,7],cmap='seismic')
s[1,3].imshow(w[0][:,:,8],cmap='seismic')
s[1,4].imshow(w[0][:,:,9],cmap='seismic')
p.savefig('Weights Step 0')

p,s = plt.subplots(2,5)
s[0,0].imshow(w[-1][:,:,0],cmap='seismic')
s[0,1].imshow(w[-1][:,:,1],cmap='seismic')
s[0,2].imshow(w[-1][:,:,2],cmap='seismic')
s[0,3].imshow(w[-1][:,:,3],cmap='seismic')
s[0,4].imshow(w[-1][:,:,4],cmap='seismic')
s[1,0].imshow(w[-1][:,:,5],cmap='seismic')
s[1,1].imshow(w[-1][:,:,6],cmap='seismic')
s[1,2].imshow(w[-1][:,:,7],cmap='seismic')
s[1,3].imshow(w[-1][:,:,8],cmap='seismic')
s[1,4].imshow(w[-1][:,:,9],cmap='seismic')
p.savefig('Weights end')
p,s = plt.subplots(2,6)
s[0,0].imshow(w[-1][:,:,2],cmap='seismic')
s[0,0].set_title('Seismic')
s[0,1].imshow(w[-1][:,:,2],cmap='PiYG')
s[0,1].set_title('PiYG')
s[0,2].imshow(w[-1][:,:,2],cmap='PRGn')
s[0,2].set_title('PRGn')
s[0,3].imshow(w[-1][:,:,2],cmap='PuOr')
s[0,3].set_title('PuOr')
s[0,4].imshow(w[-1][:,:,2],cmap='RdBu')
s[0,4].set_title('RdBu')
s[1,0].imshow(w[-1][:,:,2],cmap='RdGy')
s[1,0].set_title('RdGy')
s[1,1].imshow(w[-1][:,:,2],cmap='RdYlBu')
s[1,1].set_title('RdYlBu')
s[1,2].imshow(w[-1][:,:,2],cmap='RdYlGn')
s[1,2].set_title('RdYlGn')
s[1,3].imshow(w[-1][:,:,2],cmap='Spectral')
s[1,3].set_title('Spectral')
s[1,4].imshow(w[-1][:,:,2],cmap='coolwarm')
s[1,4].set_title('coolwarrm')
s[0,5].imshow(w[-1][:,:,2],cmap='BrBG')
s[0,5].set_title('BrBg')
s[1,5].imshow(w[-1][:,:,2],cmap='bwr')
s[1,5].set_title('bwr')
p.savefig('Diverging Colormaps')
        #ä print('Training Accuracy: ', session.run(perf,feed_dict={pic: mnist.train.images, labels: mnist.train.labels}))

# In[82]:




# In[ ]:




# In[ ]:



