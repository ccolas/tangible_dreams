import jax.numpy as jnp
import numpy as np
from jax import jit
import networkx as nx

# =====================
# Activation Functions
# =====================

@jit
def weierstrass(x):
    a = 0.5
    b = 3.0
    result = jnp.zeros_like(x)
    for i in range(4):
        result += a ** i * jnp.cos(b ** i * x + jnp.pi / 3)
    return result * 0.4

@jit
def takagi(x):
    x *= 0.5
    w = 0.6
    n = 4
    i_vals = jnp.arange(n, dtype=jnp.float32).reshape(-1, 1)
    weights = w ** i_vals
    scales = 2.0 ** i_vals
    scaled_x = scales * x
    terms = weights * jnp.abs(scaled_x - jnp.round(scaled_x))
    return jnp.sum(terms, axis=0) - 0.5

@jit
def riemann(x):
    x *= 3.0
    i_vals = jnp.arange(1.0, 5.0).reshape(-1, 1)
    squared_i = i_vals * i_vals
    terms = (1.0 / squared_i) * jnp.sin(squared_i * x)
    return jnp.sum(terms, axis=0) * 0.7

activation_fns = [
    lambda x: jnp.tanh(x * 3),
    lambda x: jnp.sin(x) + jnp.sin(2.4 * x),
    lambda x: jnp.tanh(x * 3),
    lambda x: jnp.sin(x) + jnp.sin(2.4 * x),
    lambda x: x,
    lambda x: jnp.cos(x * 2.3),
    # lambda x: weierstrass(x * 2.0),
    # lambda x: riemann(x),
    # lambda x: takagi(x),
]

# =====================
# Coordinate Transforms
# =====================

def build_coordinate_grid(res, factor):
    x = jnp.linspace(-1, 1, int(res * factor))
    y = jnp.linspace(-1, 1, res)
    return jnp.meshgrid(x, y)

def define_coordinate_inputs(x, y):
    r = jnp.sqrt(x ** 2 + y ** 2)
    theta = jnp.arctan2(y, x)

    input_list = [
        x,
        y,
        jnp.abs(x),
        jnp.abs(y),
        r / np.sqrt(2),                                   # dist
        theta / jnp.pi,                                   # angle
        jnp.sin(2 * r),                                   # radial_wave
        jnp.sin(x * np.pi) * jnp.sin(y * np.pi),          # grid
        jnp.mod(r * 2, 1.0) * 2 - 1,                       # r_mod
        jnp.sin(theta + 5 * r),                            # spiral
        jnp.sin(10 * (x ** 2 + y ** 2)),                  # ripple
        jnp.sin(theta * 5),                                # star
    ]

    return jnp.stack(input_list)  # shape: (n_inputs, H, W)

def generate_inputs(res, factor, zoom, x_offset, y_offset, selected_keys=None):
    x = jnp.linspace(-1, 1, int(res * factor))
    y = jnp.linspace(-1, 1, res)
    X, Y = jnp.meshgrid(x, y)

    X = (X / zoom) + x_offset
    Y = (Y / zoom) + y_offset

    coord_inputs = define_coordinate_inputs(X, Y)

    if selected_keys is None:
        selected_keys = ['abs_x', 'dist']

    selected_inputs = [coord_inputs[k] for k in selected_keys]
    return jnp.stack(selected_inputs).reshape(len(selected_inputs), -1)

# =====================
# Graph Connectivity
# =====================

def graph_has_path(adj_matrix, input_ids, output_ids):
    G = nx.Graph(adj_matrix)
    for i in input_ids:
        for j in output_ids:
            if nx.has_path(G, i, j):
                return True
    return False
