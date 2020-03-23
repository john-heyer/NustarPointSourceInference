import jax.numpy as np
from jax import grad, jit, vmap, random
import matplotlib.pyplot as plt

from timeit import default_timer as timer


NUSTAR_IMAGE_LENGTH = 64
PSF_IMAGE_LENGTH = 1300

# In radians/pixel
NUSTAR_PIXEL_SIZE =  5.5450564776903175e-05
PSF_PIXEL_SIZE = 2.9793119397393605e-06

PSF_HALF_LENGTH = NUSTAR_IMAGE_LENGTH/2

PSF = np.load("psf_test.npy")
PSF = np.array(PSF)
print(np.max(PSF))
print(np.min(PSF))
print(np.take(PSF, 10000))
print(np.take(PSF, -10235325))
print(np.take(PSF, 0))
print

def pixel_psf_powerlaw(i, j, source_x, source_y, psf=PSF):
	# d = (
	# 	((PSF_HALF_LENGTH - i) - source_y/NUSTAR_PIXEL_SIZE)**2 +
	# 	((j - PSF_HALF_LENGTH) - source_x/NUSTAR_PIXEL_SIZE)**2
	# )
	def clip(x):
		return np.max([0, np.min([x, NUSTAR_IMAGE_LENGTH])])
	t_x, t_y = source_x/NUSTAR_PIXEL_SIZE, source_y/NUSTAR_PIXEL_SIZE
	i_x, i_y = clip((j - t_x).astype(int)), clip((i + t_y).astype(int))
	# i = i_y*64 + i_x
	pixel = PSF[i_y,i_x]#np.take(PSF, i)
	return pixel #1/(1 + .1*d**2)


image_psf_power_law = vmap(vmap(pixel_psf_powerlaw, in_axes=(None, 0, None, None)), in_axes=(0, None, None, None))
all_sources_psf = vmap(image_psf_power_law, in_axes=(None, None, 0, 0))

def powerlaw_psf(sources_x, sources_y, sources_b):
	psf_all = all_sources_psf(np.linspace(.5, 63.5, 64), np.linspace(0.5, 63.5, 64), sources_x, sources_y)
	psf_all = psf_all / np.sum(psf_all, axis=(1,2), keepdims=True)  # Normalize
	psf_all = psf_all * sources_b[:, np.newaxis, np.newaxis]  # Scale by brightness of sources
	emission_map = np.sum(psf_all, axis=0)
	return emission_map

key = random.PRNGKey(68)
start = None
for i in range(2):
	if i == 1:
		start = timer()
	n = 195 + i%10
	key, key1, key2, key3 = random.split(key, 4)
	sources_x = NUSTAR_PIXEL_SIZE * random.uniform(key1, shape=(n,), minval=-30, maxval=30)
	sources_y = NUSTAR_PIXEL_SIZE * random.uniform(key2, shape=(n,), minval=-30, maxval=30)
	sources_b = random.uniform(key3, shape=(n,), minval=5, maxval=1000)
	psf = powerlaw_psf(sources_x, sources_y, sources_b)
end = timer()
print("time elapsed:", end - start)

jit_psf = jit(powerlaw_psf)
n = 200
for i in range(10001):
	if i == 1:
		start = timer()
	if i%100 == 0:
		u = random.uniform(key)
		if u > .5:
			n += 1
		else:
			n -= 1
	key, key1, key2, key3 = random.split(key, 4)
	sources_x = NUSTAR_PIXEL_SIZE * random.uniform(key1, shape=(n,), minval=-30, maxval=30)
	sources_y = NUSTAR_PIXEL_SIZE * random.uniform(key2, shape=(n,), minval=-30, maxval=30)
	sources_b = random.uniform(key3, shape=(n,), minval=5, maxval=1000)
	psf = jit_psf(sources_x, sources_y, sources_b)
end = timer()
print("time elapsed jit:", end - start)

# psf = image_psf_power_law(np.linspace(.5, 63.5, 64), np.linspace(0.5, 63.5, 64), 32*NUSTAR_PIXEL_SIZE, 32*NUSTAR_PIXEL_SIZE)
# # psf2 = image_psf_power_law(np.linspace(.5, 63.5, 64), np.linspace(0.5, 63.5, 64), 20.7*NUSTAR_PIXEL_SIZE, 20.7*NUSTAR_PIXEL_SIZE)
# # print(np.sum((psf-psf2)**2))
# plt.matshow(psf)
# plt.show()
