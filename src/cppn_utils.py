import jax.numpy as jnp
import jax
from src.jax_vnoise import jax_vnoise

# =====================
# Activation Functions
# =====================

noise = jax_vnoise.Noise()

activation_fns = dict(
    tanh=lambda x: jnp.tanh(x / 10),                                      # tanh
    cos=lambda x: jnp.cos(x * 3),         # cos
    leaky_relu=lambda x: jnp.where(x > 0, x, 0.1 * x),
    gaussian=lambda x: jnp.exp(-(x/2) ** 2 / 1),  # gaussian
    modulo=lambda x: ((x / 5) % 1.0) * 2 - 1,            # modulo
    riemann=lambda x: sum([1 / i**2 * jnp.sin(i**2 * x ) for i in range(1, 4)]),  # riemann
    perlin=lambda x: 2 * noise.noise1((1 * x).flatten(), octaves=3, persistence=0.1, lacunarity=5).reshape(x.shape), # perlin
    mysterious=lambda x: jax.lax.fori_loop(
        0, 5,
        lambda i, v: 4 * v * (1 - v),
        (x / 10 % 1.0)
    )

    # lambda x: sum([0.7**i * jnp.abs(2**i * jnp.cos(x) - jnp.round(2**i * jnp.cos(x / 3 / 1.5 * 3)))
    #          for i in range(3)]),  # takagi
    # lambda x: 0.5 * sum([0.9**i * jnp.cos(6**i * jnp.pi * (x/3)) for i in range(2)]), # weierstrass
    #     lambda x: 1 - jnp.exp(-jnp.exp(x * 5)),     # comp_loglog
    # lambda x: jnp.where(x > 0, 1.0, 0.0),         # step
    # lambda x: jnp.abs(x * 2),                   # abs
)

# =====================
# Coordinate Transforms
# =====================
SCALE = 10
def build_coordinate_grid(res, factor, scale=SCALE):
    x = jnp.linspace(-scale*factor, scale*factor, int(res*factor))
    y = jnp.linspace(-scale, scale, res)
    return jnp.meshgrid(x, y)

# --- Rotated Abs ---
def rotated_abs(x, y, rot_deg_raw, gain_raw):
    rot_deg = (rot_deg_raw / 1023) * 180 - 90
    gain =  (gain_raw / 1023) * 10
    rot_rad = jnp.deg2rad(rot_deg)
    projection = x * jnp.cos(rot_rad) + y * jnp.sin(rot_rad)
    return jnp.abs(projection * gain)

# --- Radial Distance ---
def generalized_radial(x, y, freq_raw, width_raw):
    r = jnp.sqrt(x**2 + y**2) / SCALE
    freq = 2 ** ((freq_raw / 1023) * 4)  # 1 to 8 oscillations
    width = (width_raw / 1023) * 10 + 1
    value = jnp.sin(r * freq) * jnp.exp(-r * width)
    return value * SCALE

# --- Angular Sinusoid ---
def angular_sinusoid(x, y, phase_raw, amplitude_raw):
    phase = (phase_raw / 1023) * 2 * jnp.pi
    amplitude = (amplitude_raw / 1023) * 4 + 1  # 0 to 2
    theta = jnp.arctan2(y, x)
    return jnp.sin(theta + phase) * amplitude

# --- Grid ---
def grid(x, y, freq_raw, alpha_raw):
    freq = 2 ** ((freq_raw / 1023) * 4 - 1)
    alpha = alpha_raw / 1023

    # Scale down coordinates to reduce grid density
    scale = 0.2  # or 0.3, adjust to taste
    x_scaled = x * scale
    y_scaled = y * scale

    return jnp.sin(jnp.pi * x_scaled * freq + 0.3 * ((1 - alpha) + alpha * jnp.sin(9 * y_scaled))) * \
        jnp.sin(jnp.pi * y_scaled * freq + 0.3 * ((1 - alpha) + alpha * jnp.sin(9 * x_scaled))) * 5

# --- Spiral Symmetry ---
def spiral_symmetry(x, y, n_folds_raw, spiral_raw):
    n_folds = jnp.round((n_folds_raw / 1023) * 7 + 2)
    spiral = jnp.tanh((spiral_raw / 1023) * 2) * 3
    theta = jnp.arctan2(y, x)
    r = jnp.sqrt(x**2 + y**2) / jnp.sqrt(2)
    return jnp.sin(jnp.mod(theta, 2 * jnp.pi / n_folds) * n_folds + spiral * r)


input_functions = dict(x=lambda x, y, p1, p2: x,
                       y=lambda x, y, p1, p2: y,
                       symmetry=lambda x, y, p1, p2: rotated_abs(x, y, rot_deg_raw=p1, gain_raw=p2),
                       radial=lambda x, y, p1, p2: generalized_radial(x, y, freq_raw=p1, width_raw=p2),
                       angular=lambda x, y, p1, p2: angular_sinusoid(x, y, phase_raw=p1, amplitude_raw=p2),
                       grid=lambda x, y, p1, p2: grid(x, y, freq_raw=p1, alpha_raw=p2),
                       spiral=lambda x, y, p1, p2: spiral_symmetry(x, y, n_folds_raw=p1, spiral_raw=p2),
                       unif=lambda x, y, p1, p2: jnp.full_like(x, ((p1 / 1023.0) - 0.5) * 6))
input_selector_mapping = ['x', 'y', 'symmetry', 'radial', 'angular', 'grid', 'spiral', 'unif']


# param mapping

# Mapping functions
def zoom_mapping(value, vmin=0, vmax=1023, log_range=2.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    return 10 ** (norm * log_range)

def bias_mapping(value, vmin=0, vmax=1023, range_=10.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    scaled = jnp.sign(norm) * (jnp.exp(jnp.abs(norm) * 3) - 1) / (jnp.exp(3) - 1)
    return range_ * scaled

def norm_value(value, magnitude):
    return value / 1023 - 0.5 * 2 * magnitude

def weight_mapping(value, vmin=0, vmax=1023, range_=5.0):
    # Normalize knob: [0..1023] → [-1..1]
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)

    # Smooth nonlinear scaling (like your bias/contrast)
    scaled = jnp.sign(norm) * (jnp.exp(jnp.abs(norm) * 3) - 1) / (jnp.exp(3) - 1)

    # Map: center=1, left→-range_, right→+range_
    return 1 + scaled * (range_ - 1)



def slope_mapping(value, vmin=0, vmax=1023, min_c=0.0, max_c=5.0, mid=1.0, exp_k=3.0):
    norm = (value - vmin) / (vmax - vmin)
    if norm <= 0.5:
        t = norm / 0.5
        scaled = (jnp.exp(t*exp_k)-1) / (jnp.exp(exp_k)-1)
        return min_c + (mid - min_c) * scaled
    else:
        t = (norm - 0.5) / 0.5
        scaled = (jnp.exp(t*exp_k)-1) / (jnp.exp(exp_k)-1)
        return mid + (max_c - mid) * scaled

def contrast_mapping(value, vmin=0, vmax=1023, min_c=0.0, max_c=5.0, mid=1.0, exp_k=3.0):
    norm = (value - vmin) / (vmax - vmin)
    if norm <= 0.5:
        t = norm / 0.5
        scaled = (jnp.exp(t*exp_k)-1) / (jnp.exp(exp_k)-1)
        return min_c + (mid - min_c) * scaled
    else:
        t = (norm - 0.5) / 0.5
        scaled = (jnp.exp(t*exp_k)-1) / (jnp.exp(exp_k)-1)
        return mid + (max_c - mid) * scaled


def balance_mapping(value, vmin=0, vmax=1023, range_=5.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    scaled = jnp.sign(norm) * (jnp.exp(jnp.abs(norm) * 3) - 1) / (jnp.exp(3) - 1)
    return range_ * scaled

def mods_mapping(value, vmin=0, vmax=1023, range_=5.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    scaled = jnp.sign(norm) * (jnp.exp(jnp.abs(norm) * 3) - 1) / (jnp.exp(3) - 1)
    return range_ * scaled