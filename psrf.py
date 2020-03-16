import numpy as np
import matplotlib.pyplot as plt


def compute_psrf(emission_maps):
	"""
	Computes the potential scale reduction factor, for each component of the
	emission map produced by the sources, across M chains with N samples each.
	Thus, emission_maps is a tensor of shape (N, M, 64, 64)
	Returns: psrf_map of shape (64, 64)
	"""
	N, M = emission_maps.shape[0], emission_maps.shape[1]
	means_by_chain = np.mean(emission_maps, axis=0)  # (M, 64, 64)
	mean_of_means = np.mean(means_by_chain, axis=0)  # (64, 64)
	B = N/(M - 1) * np.sum((means_by_chain - mean_of_means) ** 2, axis=0)  # (64, 64)
	variances_by_chain = 1/(N - 1) * np.sum((emission_maps - means_by_chain) ** 2, axis=0)  # (M, 64, 64)
	W = np.mean(variances_by_chain, axis=0)  # (64, 64)
	variance_estimates = (N - 1)/N * W + 1/N * B
	R_hat = np.sqrt(variance_estimates / W)  # (64, 64)
	return R_hat



emission_maps = np.random.rand(500, 16, 64, 64)


psrf = compute_psrf(emission_maps)
psrf = psrf.reshape(64*64)
psrf = sorted(psrf)
plt.hist(psrf, bins=60, ec='black')
plt.show()