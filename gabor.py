
import numpy as np
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


def filterImageSet(images,wavelength):
    
    # Filter characteristics
    psi = np.array([0,np.pi,-np.pi/2,np.pi/2])
    orient = np.array([0,np.pi/4,np.pi/3,3*np.pi/4])
    bw = 1.5
    gamma = 0.5
    imSize = images.shape[1]
    if images.shape[1] != images.shape[2]:
        raise ValueError('Images need to be squared')
    
    # output Matrix
    All = np.zeros([images.shape[0],images.shape[1],images.shape[2],16])
    OUT = np.zeros(images.shape)
    
    # Create Gabor Filter
    gabor = np.zeros([len(psi)*len(orient),7,7])
    filtNum = 0
    for p in psi:
        for o in orient:
            gabor[filtNum,:,:] = createGabor(bw,gamma,p,wavelength,o)
            filtNum += 1

    
    # Filter each image
    for imgNum in range(images.shape[0]):
        img = images[imgNum,:,:]
        paddedImage = np.zeros([2*imSize,2*imSize])
        top = 1/2 * imSize
        left = top # since images are squared
        x = np.arange(left,2*imSize-left)
        paddedImage[left:2*imSize-left , top:2*imSize-top] = img
        # paddedImage[np.arange(left,2*imSize-left).astype(int), np.arange(top,2*imSize-top).astype(int)] = img
        filtNum = 0
        for p in psi:
            for o in orient:
                tmp = signal.convolve2d(paddedImage,gabor[filtNum,:,:],boundary='symm',mode='same')
                tmp = tmp[left:2*imSize-left, top:2*imSize-top] # Crop 
                # Use full range
                All[imgNum,:,:,filtNum] = tmp
                filtNum += 1
        # Merge            
        OUT[imgNum,:,:] = abs(np.mean(All[imgNum,:,:,:],axis=2))
        OUT[imgNum,:,:] = (OUT[imgNum,:,:] - np.min(OUT[imgNum,:,:])) / (np.max(OUT[imgNum,:,:]-np.min(OUT[imgNum,:,:])))
    return OUT        
        
        
def createGabor(bw,gamma,psi,wl,theta):
    
    sigma = (wl/np.pi) * np.sqrt(np.log(2)/2) * (np.power(2,bw)+1)/(np.power(2,bw)-1)
    sigma_x = sigma
    sigma_y = sigma/gamma
    
    sz = np.fix(8*max(sigma_x,sigma_y))
    sz += 1 if sz%2 == 0 else 0 # Adjust to even filter size 
    [a,b] = np.meshgrid(np.arange(-np.fix(sz/2),np.fix(sz/2)+1), np.arange(np.fix(sz/2),np.fix(-sz/2)-1,-1))
    
    # Rotate
    a_theta = a*np.cos(theta) + b*np.sin(theta)
    b_theta = -a*np.sin(theta) + b*np.cos(theta)
    
    e = np.exp(-0.5 * (np.power(a_theta,2)/np.power(sigma_x,2) + np.power(b_theta,2)/np.power(sigma_y,2)))
    f = np.cos(2*np.pi/wl*a_theta + psi)
    gabor = e*f
    # Set mean to 0 (compensate for numerical irrregularities)
    gabor -= np.mean(np.mean(gabor))
    # Normalize
    gabor = gabor / np.linalg.norm(gabor)
    return gabor