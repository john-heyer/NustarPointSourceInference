import jax
import jax.numpy as np
from jax import pmap, random, jit
from timeit import default_timer as timer


print(jax.local_device_count())  # 4

def random_walk(key, steps=1000):
	position = 0
	for _ in range(steps):
		key, subkey = random.split(key)
		position += random.normal(subkey)
	return position

jit_random_walk = jit(random_walk)

p_random_walk = pmap(jit_random_walk)


start = timer()
jit_random_walk(random.PRNGKey(0))
end = timer()
print("compile time serial:", end - start)

start = timer()
for i in range(4):
	jit_random_walk(random.PRNGKey(i))
end = timer()
print("time elapsed serial:",end - start)


keys = np.array([random.PRNGKey(i) for i in range(4)])

start = timer()
p_random_walk(keys)
end = timer()
print("compile time parallel:", end - start)

start = timer()
p_final_positions = p_random_walk(keys)
end = timer()
print("time elapsed parallel:", end - start)

