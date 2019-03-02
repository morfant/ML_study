import numpy as np
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('/Users/giy/Pictures/jinnam_sleep.jpg')

plt.imshow(img)
plt.show()