import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os


PSF_PIXEL_SIZE = 2.9793119397393605e-06
window_min, window_max = -1300/2, 1300/2

zero_data = np.load("zero_move.npz")

new = zero_data["new"]
head = zero_data["head"]

n_map = zero_data["n_map"]
o_map = zero_data["o_map"]
sample = zero_data["sample"]

n_x, n_y, n_b = new[0]/PSF_PIXEL_SIZE, new[1]/PSF_PIXEL_SIZE, np.exp(new[2])
i_x, i_y, i_b = head[0]/PSF_PIXEL_SIZE, head[1]/PSF_PIXEL_SIZE, np.exp(head[2])


plt.scatter(x=n_x, y=n_y, c=n_b, s=10, edgecolors='black')
plt.scatter(x=i_x, y=i_y, c=i_b, s=10, edgecolors='black')
plt.gca().add_patch(Rectangle((window_min,window_min),2*window_max,2*window_max,linewidth=.5,edgecolor='r',facecolor='none'))
plt.gca().add_patch(Rectangle((1.1*window_min,1.1*window_min),1.1*2*window_max,1.1*2*window_max,linewidth=.5,edgecolor='black',facecolor='none'))
plt.xlim(-1200, 1200)
plt.ylim(-1200, 1200)
plt.show()


plt.matshow(n_map)
plt.matshow(o_map)
plt.matshow(sample)
plt.show()