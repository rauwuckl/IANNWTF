import numpy as np
from scipy import ndimage
def sobelImageSet(images):
    # Apply Sobel Filter in x- and y-direction.
    # Return combined sobelFilter
    sobX = np.zeros(images.shape)
    sobY = np.zeros(images.shape)
    OUT = np.zeros(images.shape)
    for img in range(images.shape[0]):
        sobX[img,:,:] = ndimage.sobel(images[img,:,:],axis=0,mode='constant')
        sobY[img,:,:] = ndimage.sobel(images[img,:,:],axis=1,mode='constant')
        OUT[img,:,:] = np.hypot(sobX[img,:,:],sobY[img,:,:])
        # scale to range [0,1]
        OUT[img,:,:] = (OUT[img,:,:]-np.min(OUT[img,:,:])) / (np.max(OUT[img,:,:])-np.min(OUT[img,:,:]))
    return OUT