import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from genetic_algorithm import GeneticAlgorithm
import utils

IMG_SIZE = 256

# read target image
img = cv.imread("apples.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (IMG_SIZE, IMG_SIZE))

ga = GeneticAlgorithm(img, num_gens=10000, init_pop=100, new_per_gen=50, mut_pct=0.25)
best_individuals = ga.run_simulation()

print('\nbest individuals of each generation...')
for i in range(len(best_individuals)):
    print(f'{i}: {best_individuals[i].fitness}')

best_img = [utils.chromosome2img(x.chromosome, (IMG_SIZE, IMG_SIZE)) for x in best_individuals]

fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(9, 6))
axs[0].imshow(img, cmap='gray')
axs[0].axis('off')
for ax, img in zip(axs[1:].flat, best_img):
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()