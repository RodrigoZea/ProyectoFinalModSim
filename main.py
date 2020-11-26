import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from genetic_algorithm import GeneticAlgorithm
import utils
from individual import Individual
from gen_record import GenRecord

IMG_SIZE = 256

# read target image
img = cv.imread("face.png", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (IMG_SIZE, IMG_SIZE))

ga = GeneticAlgorithm(img, num_gens=1000000, init_pop=20, new_per_gen=10, mut_pct=0.001, checkpoints=100)
print(f'Best fitness {ga.max_fitness}')
input('Press any key to begin simulation...')
ga_results = ga.run_simulation()

# process GA results
imgs = []
for r in ga_results:
  img = chromosome2img(r.chromosome, (IMG_SIZE, IMG_SIZE))
  imgs.append(img)
  cv.imwrite(f'./output/img/best_{r.gen_number}.png', img)
# append target image
imgs = [img] + imgs

# show image results
cols = 5
rows = len(imgs) // cols
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))
for r in range(rows):
    for c in range(cols):
        axs[r][c].imshow(imgs[r + c], cmap='gray')
        axs[r][c].axis('off')
plt.tight_layout()
plt.show()

# show how fitness was improving
xs = [x.gen_number for x in ga_results]
ys = [x.fitness for x in ga_results]
fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.set(xlabel='Número de Generación', ylabel='Fitness',
       title='Evolución de Fitness')
ax.grid()
fig.savefig("fitness.png")
plt.show()