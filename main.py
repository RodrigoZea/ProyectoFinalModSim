import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from genetic_algorithm import GeneticAlgorithm

IMG_SIZE = 512

img = cv.imread("apples.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (IMG_SIZE, IMG_SIZE))

plt.imshow(img, cmap='gray')
plt.show()