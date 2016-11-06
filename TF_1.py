
# coding: utf-8

# Import Modules

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# # The MLP

# In[33]:


class MLP:
    
    def __init__(self,*neurons):
        self.layers = len(neurons)
        self.neuPerL = [n for n in neurons]
    
    def setWeights(self,init):
        if init == 'SND':
            self.weights = [float(np.random.randn(1)) for l in range(self.layers-1)]
        elif init == 'Uniform':
            self.weights = [float(np.random.rand(1)) for l in range(self.layers-1)]
        elif init == 'LeCun':
            self.weights = [float(np.random.normal(0,1/np.sqrt(self.neuPerL[l]),1)) for l in range(self.layers-1)]
        self.lastChanges = [0 for k in self.weights]
  
              
    def tanh(self,x):
        #return 1/(1+np.exp(-x))
        return 1.7159 * np.tanh(2/3 * x)
            
    def der(self,x):
        #return np.exp(x)/(np.exp(x)+1)**2
        return 1.14393 * 1/np.cosh(2*x/3)**2
    
    def forward(self,inp):
        self.activation = [inp]
        for l in range(self.layers-1):
            self.activation.append(self.tanh(np.dot(self.activation[-1],self.weights[l])))       
             
    def sgd(self,LR,epochs,mom,data,targ):
        # Implements Stochastic Gradient Descent
        self.LR = LR
        self.mom = mom
        p = np.zeros([epochs+1])
        p[0] = self.test(data,targ)
        for epoch in range(epochs):
            for sample,target in zip(data,targ):
                self.forward(sample)   
                # Compute errors
                self.deltas = [(self.activation[-1] - target) * self.der(np.dot(self.activation[-2],self.weights[-1]))]
                for l in range(len(self.weights)-1):
                    self.deltas.append(self.der(np.dot(self.activation[-3-l],self.weights[-2-l])) * self.weights[-1-l]*self.deltas[-1-l])
                self.deltas = list(reversed(self.deltas))
                
                # Adapt weights
                self.adaptWeights()   
            #self.LR = 0.9 * self.LR # Decaying LR
            p[epoch+1] = self.test(data,targ)
        return p
            
    def batch_gd(self,LR,epochs,mom,data,targets):
        # Implements Batch Gradient Descent
        p = np.zeros([epochs+1])
        self.mom = mom
        self.LR = LR
        for epoch in range(epochs):
            activity = np.zeros([len(data),self.layers])
            for ind,sample in enumerate(data):
                self.forward(sample) 
                activity[ind,:] = self.activation
            loss = np.mean(activity[:,-1] - targets) 
            self.deltas = [loss * self.der(np.mean(activity[:,-2])*self.weights[-1])]
            for l in range(len(self.weights)-1):
                self.deltas.append(self.der(np.mean(activity[:,-3-l])*self.weights[-2-l]) * self.weights[-1-l]*self.deltas[-1-l])
            #self.adaptWeights()
            p[epoch+1] = self.test(data,targets)
        return p
    
    def adaptWeights(self):
        for l in range(len(self.weights)):
            tmp =  self.LR * self.deltas[l] * self.activation[l] + self.mom * self.lastChanges[l]
            self.weights[l] = self.weights[l] - tmp
            self.lastChanges[l] = tmp

    def test(self,data,targ):
        correct = 0
        for sample,target in zip(data,targ):
            self.forward(sample)
            correct += 1 if round(self.activation[-1]) == target else 0
        return 100*correct/len(data)



# Generate and normalize Train Data

# In[3]:

sampleSize = 30
np.random.seed(1)
cats = np.random.normal(25,5,sampleSize)
dogs = np.random.normal(55,15,sampleSize)


data = np.append(cats,dogs)
data = (data-np.mean(data)) / np.std(data)
t_c = -1 * np.ones([sampleSize])
t_d = 1 * np.ones([sampleSize])
targets = np.append(t_c,t_d)


# Set Model Hyperparamter

# In[4]:

weight_init = ['Uniform','SND', 'LeCun']
LR = 0.3
epochs = 100
decay = 0.2


# In[13]:

# Call this Method to evaluate the performance of the network
# Call it with True adds Momentum, calling it with False leaves it out
def evaluate(mom):
    perf = np.zeros([2*len(weight_init),epochs+1])
    for ind,w in enumerate(weight_init):
        # SGD
        net = MLP(1,1,1)
        net.setWeights(w)
        perf[ind,:] = net.sgd(LR,epochs,mom,data,targets)
        plt.plot(np.arange(perf.shape[1]),perf[ind,:],label=['SGD',w])

        # Batch
        net = MLP(1,1,1)
        net.setWeights(w)
        perf[ind+len(weight_init),:] = net.batch_gd(LR,epochs,mom,data,targets)
        plt.plot(np.arange(perf.shape[1]),perf[ind+len(weight_init),:],label=['Batch',w])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),ncol=3, fancybox=True, shadow=True)
    plt.show()


# In[32]:

evaluate(0) # Evaluate without Momentum


# In[25]:

evaluate(decay) # Evaluate with Momentum


# In[ ]:



