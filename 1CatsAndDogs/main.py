import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MLP:
	def __init__(self):
		self.w1=4*np.random.rand()-2
		self.w2=4*np.random.rand()-2
		self.w1=4
		self.w2=-4
		self.epsilon = 0.2
		self.nLayer=2
		self.dh = DataHandle()
		self.w1History =[self.w1]
		self.w2History =[self.w2]
	

	def activation(self, x):
		return 1.7159*np.tanh(x*2./3.)
	def activationStrich(self, x):
		return (4.57572*np.cosh(2./3.*x)**2)/((np.cosh(1.3333*x)+1)**2)

	def forwardStep(self, x1):
		x2 = self.activation(x1*self.w1)
		x3 = self.activation(x2*self.w2)
		return (x1, x2, x3)

	def backwardStep(self,activations , target):
		(x1, x2, x3) = activations	
		sigma2 = (x3-target)*self.activationStrich(x3)
		deltaW2 = x2 * sigma2
		
		sigma1 = self.activationStrich(x2) * self.w2 * sigma2
		deltaW1 = x1 * sigma1 
		
		return (deltaW1, deltaW2)
	
	def batchGD(self, N):
		data, label = self.dh.returnLabledBoth()
		tData, tLabel = self.dh.getNormalizedTestData(10)
		for i in range(N):
			self.GD(data, label)
			mse = self.getMSE(data, label)
			print("at Epoch {} we have a training error of {}".format(i, mse))
			

	def stochasticGD(self, N):
		data, label = self.dh.returnLabledBoth()
		for i in range(N):
			data, label = self.dh.getNextSample()
			self.GD(data, label)
		
		mse = self.getMSE(data, label)
		print("at Epoch {} we have a training error of {}".format(i, mse))
		tData, tLabel = self.dh.getNormalizedTestData(50)
		mse = self.getMSE(tData, tLabel)
		print("at Epoch {} we have a test error of {}".format(i, mse))
		
		

	def GD(self, data, target):
		activations = self.forwardStep(data) # data is a vector 
		# print("activations: {}".format(activations))
		# activations will be a tripple of vectors
		deltaW1, deltaW2 = self.backwardStep(activations, target) 
		# print("deltaW1: {}".format(deltaW1))
		# a tuple of vectors with the gradient for each weight for each sample
		dW1 = np.mean(deltaW1)
		dW2 = np.mean(deltaW2)
		print("dW: ({},{})".format(dW1,dW2))
		self.w1 = self.w1 - self.epsilon *dW1
		self.w2 = self.w2 - self.epsilon *dW2
		self.w1History.append(self.w1)
		self.w2History.append(self.w2)
		
	

	def getMSE(self, data, label):
		_, _, yHat = self.forwardStep(data)
		dif = yHat-label
		return np.mean(dif**2)

class DataHandle:
	def __init__(self,sampleSize = 30):
		np.random.seed(1)
		self.cats = np.random.normal(25, 5, sampleSize)
		self.dogs = np.random.normal(45, 15, sampleSize)
		self.normalize()
		self.lastSampleWas = -1

	def getNormalizedTestData(self, N):
		cats = np.random.normal(25, 5, N)
		dogs = np.random.normal(45, 15,N)
		both = np.concatenate((cats, dogs))
		mu = np.mean(both)
		sigma = np.std(both)

		both = (both-mu)/sigma
		label = np.concatenate((-1*np.ones(N), np.ones(N)))
		return both, label
		
		
	def getNextSample(self):
		if(self.lastSampleWas > 0):
			self.lastSampleWas = -1
			return np.random.choice(self.cats), -1
		else:
			self.lastSampleWas = 1
			return np.random.choice(self.dogs), 1
			
		
		

	def normalize(self):
		both = np.concatenate((self.cats, self.dogs))
		mu = np.mean(both)
		sigma = np.std(both)

		self.cats = (self.cats-mu)/sigma
		self.dogs = (self.dogs-mu)/sigma

	def returnLabledBoth(self):
		both = np.concatenate((self.cats, self.dogs))
		label = np.concatenate((-1*np.ones(len(self.cats)), np.ones(len(self.dogs))))
		return both, label

		

# target value -1 cat; +1 dog

def plotErrorSurface(density, wH1, wH2):
	mlp = MLP()
	dh = DataHandle()
	data, label = dh.returnLabledBoth()
	weightRange = np.linspace(-4, 4, density)
	W1, W2 = np.meshgrid(weightRange, weightRange)
	allMse = np.zeros(np.shape(W1))
	
	for i in range(density):
		for j in range(density):
			mlp.w1=W1[i,j]	
			mlp.w2=W2[i,j]
			mse = mlp.getMSE(data, label)
			allMse[i,j] = mse
	
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	#cp = ax.plot_surface(np.reshape(allW1,(density, density)),np.reshape(allW2, (density, density)), np.reshape(allMse, (density, density)), cmap = plt.cm.coolwarm)
	cp = ax.plot_surface(W1, W2, allMse, cmap = plt.cm.coolwarm)
	plt.colorbar(cp)
	plt.show()
	################
	plt.figure()
	cp = plt.contourf(W1, W2, allMse)
	plt.plot(wH1,wH2,c='black')
	plt.colorbar(cp)
	plt.show()


	return W1, W2, allMse

		
		
		

	
		
plt.close("all")
mlp = MLP()	
	
