import jax.numpy as jnp
import numpy as np
from jax import jit
import networkx as nx
from src.jax_vnoise import jax_vnoise

# =====================
# Activation Functions
# =====================

noise = jax_vnoise.Noise()

activation_fns = [
    lambda x: jnp.tanh(x * 3),                                      # tanh
    lambda x: jnp.cos(x * 3),         # cos
    lambda x: x * 5.0 / (1 + jnp.exp(-x * 5.0)),  # swish
    lambda x: jnp.exp(-x ** 2 / 1),  # gaussian
    # lambda x: ((x / 5) % 1.0) * 2 - 1,            # modulo
    # lambda x: sum([1 / i**2 * jnp.sin(i**2 * x ) for i in range(1, 4)]),  # riemann
    lambda x: 2 * noise.noise1((1 * x).flatten(), octaves=3, persistence=0.1, lacunarity=5).reshape(x.shape), # perlin
    # lambda x: sum([0.7**i * jnp.abs(2**i * jnp.cos(x) - jnp.round(2**i * jnp.cos(x / 3 / 1.5 * 3)))
    #          for i in range(3)]),  # takagi
    # lambda x: 0.5 * sum([0.9**i * jnp.cos(6**i * jnp.pi * (x/3)) for i in range(2)]), # weierstrass
    #     lambda x: 1 - jnp.exp(-jnp.exp(x * 5)),     # comp_loglog
    # lambda x: jnp.where(x > 0, 1.0, 0.0),         # step
    # lambda x: jnp.abs(x * 2),                   # abs
]

# =====================
# Coordinate Transforms
# =====================

def build_coordinate_grid(res, factor):
    x = jnp.linspace(-1, 1, int(res * factor))
    y = jnp.linspace(-1, 1, res)
    return jnp.meshgrid(x, y)


n_folds = 6
spiral = 0.1
alpha=0
input_functions = \
[
    lambda x, y: x,
    lambda x, y: y,
    lambda x, y: jnp.abs(x),
    lambda x, y: jnp.abs(y),
    lambda x, y: jnp.sqrt(x ** 2 + y ** 2) / jnp.sqrt(2),  # dist
    lambda x, y: y / jnp.sqrt(x ** 2 + y ** 2 + 1e-6),  # sin(theta)
    lambda x, y: jnp.exp(-((x ** 2 + y ** 2 - 0.5) ** 2) * 20),  # ring
    lambda x, y: jnp.sin(jnp.pi * x * 3 + 0.3 * ((1 - alpha) + alpha * jnp.sin(9 * y))) * jnp.sin(jnp.pi * y * 3 + 0.3 * ((1-alpha) + alpha * jnp.sin(9 * x))),  # grid
    lambda x, y: jnp.sin(jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi / n_folds) * n_folds + spiral  * jnp.sqrt(x ** 2 + y ** 2) / jnp.sqrt(2))  # spiral (n_folders, rotation)
]

