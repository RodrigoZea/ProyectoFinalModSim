import numpy as np
import math

def img2chromosome(img, grayscale=True):
    """Convert image to chromosome (1D array).
    
    Returns the image as a numpy 1d array.
    """
    if grayscale:
        return np.reshape(img, img.shape[0] * img.shape[1])
    return np.reshape(img, img.shape[0] * img.shape[1] * img.shape[2])

def chromosome2img(chrom, img_shape):
    return np.reshape(chrom, img_shape)

def random_img(img_size=512, grayscale=True):
    if grayscale:
        return np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)
    return np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)