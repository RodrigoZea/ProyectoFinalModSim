import numpy as np
import math

def img2chromosome(img):
    """Convert image to chromosome (1D array).
    
    Returns the image as a numpy 1d array.
    """

    return np.reshape(img, img.shape[0] * img.shape[1])

def chromosome2img(chrom, img_shape):
    """Convert chromosome to image.
    """
    return np.reshape(chrom, img_shape)

def random_img(img_size=512):
    """Create a random noise-like image.
    """
    return np.random.randint(0, 256, (img_size, img_size), dtype=np.uint8)