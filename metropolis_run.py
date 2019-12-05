import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PSF_PIXEL_SIZE = 2.9793119397393605e-06
window_min, window_max = -1300/2, 1300/2

metropolis_data = np.load("metropolis_data.npz")

posterior = metropolis_data["posterior"]
ground_truth = metropolis_data["gt"]
init = metropolis_data["init"]

with open("acceptance_stats.json", "r") as f:
	stats = json.load(f)
n_sources_counts = stats.pop('n_sources_counts')

print("\n======acceptance stats======")
for stat in stats:
	print(stat, ":", stats[stat])
print("============================\n")


print("n sources init: ", init.shape[1])
print("n sources gt: ", ground_truth.shape[1])
print("======n source counts======")
for count in n_sources_counts:
	print(count, ":", n_sources_counts[count])
print("===========================\n")

# print(ground_truth.shape)
# print(posterior.shape)

print(max(ground_truth[0]), min(ground_truth[0]))
print(max(ground_truth[1]), min(ground_truth[1]))
print(max(ground_truth[2]), min(ground_truth[2]))
print(max(posterior[0]), min(posterior[0]))
print(max(posterior[1]), min(posterior[1]))
print(max(posterior[2]), min(posterior[2]))


gt_x, gt_y, gt_b = ground_truth[0]/PSF_PIXEL_SIZE, ground_truth[1]/PSF_PIXEL_SIZE, np.exp(ground_truth[2])
i_x, i_y, i_b = init[0]/PSF_PIXEL_SIZE, init[1]/PSF_PIXEL_SIZE, np.exp(init[2])

p_x, p_y, p_b = posterior[0]/PSF_PIXEL_SIZE, posterior[1]/PSF_PIXEL_SIZE, np.exp(posterior[2])


# print(p_x.shape, p_y.shape, p_b.shape)

plt.hist2d(x=p_x, y=p_y, range=[[-1200, 1200], [-1200, 1200]], bins=128)#, weights=p_b)
plt.scatter(x=gt_x, y=gt_y, c=gt_b, s=10, edgecolors='black')
plt.scatter(x=i_x, y=i_y, s=10, c='r', edgecolors='black')
plt.gca().add_patch(Rectangle((window_min,window_min),2*window_max,2*window_max,linewidth=.5,edgecolor='r',facecolor='none'))
plt.gca().add_patch(Rectangle((1.5*window_min,1.5*window_min),1.5*2*window_max,1.5*2*window_max,linewidth=.5,edgecolor='black',facecolor='none'))
plt.show()

# X = np.array([p_x, p_y]).T
# init = np.array([gt_x, gt_y]).T
# print(X.shape, init.shape)
# kmeans = KMeans(n_clusters=ground_truth.shape[1], random_state=0).fit(X)

# print(max(p_x), max(p_y))
# print(min(p_x), min(p_y))
# centers = kmeans.cluster_centers_
# print(init)
# print(centers)

# maps = np.load("inf_ratio.npz")
# sample_map = maps["sampled_img"]
#
#
# accepted = maps["accepted_img"]
# previous = maps["previous_img"]
#
# plt.matshow(previous)
#
# plt.show()
