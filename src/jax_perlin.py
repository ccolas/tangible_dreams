import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def hash_noise(seed, x):
    x = (x ^ seed) * 0x27d4eb2d
    x = (x ^ (x >> 15)) * 0x27d4eb2d
    x = x ^ (x >> 15)
    return jnp.sin(x * 12.9898 + seed) * 43758.5453 % 2 - 1

@jax.jit
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

@jax.jit
def perlin_1d(x, seed):
    xi = jnp.floor(x).astype(int)
    xf = x - xi

    a = hash_noise(seed, xi)
    b = hash_noise(seed, xi + 1)

    u = fade(xf)
    return (1 - u) * a + u * b


@partial(jax.jit, static_argnames=['octaves'])
def perlin_1d_fractal(x, seed=10, octaves=4, freq=1.0, amp=1.0, gain=0.5, lacunarity=3.0):
    indices = jnp.arange(octaves)
    def octave(i):
        f = freq * lacunarity ** i
        a = amp * gain ** i
        return perlin_1d(x * f, seed + i) * a

    indices = jnp.arange(octaves)
    values = jax.vmap(octave)(indices)
    return jnp.sum(values, axis=0)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    x = jnp.linspace(0, 10, 1000)
    plt.figure()
    for i in range(10):
        y = perlin_1d_fractal(x, seed=np.random.randint(1e6), octaves=3, freq=1.0, amp=1.0, gain=0.5, lacunarity=2.0)
        plt.plot(x, y, linewidth=5)
    plt.savefig("/home/cedric/Downloads/perlin.pdf")
    plt.show()