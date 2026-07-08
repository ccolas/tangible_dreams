import jax.numpy as jnp
import jax
from src.jax_vnoise import jax_vnoise

# =====================
# Activation Functions
# =====================

noise = jax_vnoise.Noise()

activation_fns = dict(
    tanh=lambda x: jnp.tanh(x),                                      # tanh
    cos=lambda x: jnp.cos(x * 3),         # cos
    leaky_relu=lambda x: jnp.clip(jnp.where(x > 0, 2 * x, 0.5 * x), -4, 4) / 3,
    gaussian=lambda x: jnp.exp(-((x * 2) ** 2)),
    modulo=lambda x: ((x) % 1.0) * 2 - 1,            # modulo
    riemann=lambda x: sum([2 / (i*2)**2 * jnp.sin(i**2 * x * 5) for i in range(1, 7)]),  # riemann
    perlin=lambda x: 2 * noise.noise1((2 * x).flatten(), octaves=5, persistence=0.1, lacunarity=10).reshape(x.shape), # perlin
    # mysterious=lambda x: jax.lax.fori_loop(
    #     0, 5,
    #     lambda i, v: 4 * v * (1 - v),
    #     (x / 10 % 1.0)
    # ),

    # 9th hardware selector position (activation switch is bounds-checked against len(activations),
    # so this slot is live). Currently active: 'bimodal' — two sharp triangular (piecewise-linear,
    # NOT gaussian) peaks with a valley between, flat elsewhere; genuinely non-periodic (no repeat),
    # not a ring, not a single smooth bump/S-curve. To try an alternative, comment this one out and
    # uncomment exactly one of the others below.
    bimodal=lambda x: jnp.maximum(jnp.clip(1 - 1.2 * jnp.abs(x - 1.2), -1, 1),
                                   jnp.clip(1 - 1.2 * jnp.abs(x + 1.2), -1, 1)),
    # wavelet=lambda x: (1 - x ** 2) * jnp.exp(-(x ** 2) / 2),   # mexican-hat wavelet: ringed bump
    # swish=lambda x: x * jax.nn.sigmoid(x),                    # smooth modern nonlinearity, nonwarp-periodic
    # fold=lambda x: jnp.abs(x * 2) - 1,                        # sharp V-fold, non-periodic (vs. modulo's sawtooth)

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
# SCALE = 10
def build_coordinate_grid(res, factor, scale=1):
    x = jnp.linspace(-scale*factor, scale*factor, int(res*factor))
    y = jnp.linspace(-scale, scale, res)
    return jnp.meshgrid(x, y)

# --- Rotated Abs ---
def rotated_abs(x, y, rot_deg_raw, gain_raw):
    rot_deg = (rot_deg_raw / 1023) * 180 - 90
    gain =  (gain_raw / 1023) * 10
    rot_rad = jnp.deg2rad(rot_deg)
    projection = x * jnp.cos(rot_rad) + y * jnp.sin(rot_rad)
    return jnp.abs(projection * gain) * 2 - 1

# --- Radial Distance ---
def generalized_radial(x, y, freq_raw, width_raw):
    r = jnp.sqrt(x**2 + y**2)
    freq = 2 ** ((freq_raw / 1023) * 4.5)  # 1 to 8 oscillations
    width = (width_raw / 1023) * 3 + 0.5
    value = jnp.sin(r * freq) * jnp.exp(-r * width)
    return value

# --- Angular Sinusoid ---
def angular_sinusoid(x, y, phase_raw, amplitude_raw):
    phase = (phase_raw / 1023) * 2 * jnp.pi
    amplitude = (amplitude_raw / 1023) + 0.4  # 0 to 2
    theta = jnp.arctan2(y, x)
    return jnp.sin(theta + phase) * amplitude

# --- Grid ---
def grid(x, y, freq_raw, alpha_raw):
    # Map knob to a useful frequency range, e.g. 1–10
    freq = 3 + (freq_raw / 1023) * 20
    alpha = alpha_raw / 1023

    # Scale coordinates so whole screen shows ~5 cells at freq=5
    # If x,y range is [-10,10], then dividing by 2 gives ~5 cells
    scale = 0.25
    x_scaled = x * scale
    y_scaled = y * scale

    return (
            jnp.sin(jnp.pi * x_scaled * freq + 1* ((1 - alpha) + alpha * jnp.sin(9 * y_scaled))) *
            jnp.sin(jnp.pi * y_scaled * freq + 1 * ((1 - alpha) + alpha * jnp.sin(9 * x_scaled)))
    )

# --- Spiral Symmetry ---
def spiral_symmetry(x, y, n_folds_raw, spiral_raw):
    n_folds = jnp.round((n_folds_raw / 1023) * 7 + 2)
    spiral = jnp.tanh((spiral_raw / 1023) * 5) * 3
    theta = jnp.arctan2(y, x)
    r = jnp.sqrt(x**2 + y**2) / jnp.sqrt(2)
    return jnp.sin(jnp.mod(theta, 2 * jnp.pi / n_folds) * n_folds + spiral * r)

# --- Wave Interference ---
def waves(x, y, angle_raw, freq_raw):
    delta = (angle_raw / 1023) * jnp.pi          # 0 to 180 degrees: parallel -> fully opposed waves
    freq = 2 ** ((freq_raw / 1023) * 4.5)         # ~1 to ~23 cycles, exponential (matches radial's freq knob feel)
    a1, a2 = -delta / 2, delta / 2
    w1 = jnp.sin((x * jnp.cos(a1) + y * jnp.sin(a1)) * freq)
    w2 = jnp.sin((x * jnp.cos(a2) + y * jnp.sin(a2)) * freq)
    return w1 * w2  # product (not sum) — a sum cancels to exactly 0 at delta=180deg, product never does

# --- Domain-Warped Coordinate ---
def domain_warp(x, y, freq_raw, strength_raw):
    freq = 0.2 + (freq_raw / 1023) * 1.5
    strength = (strength_raw / 1023) * 6.0
    n = noise.noise2(x * freq, y * freq, octaves=3, persistence=0.5, lacunarity=2.0, grid_mode=False)
    return jnp.sin(x + n * strength)

# --- Voronoi / Cellular ---
def _cell_hash(cx, cy):
    h1 = jnp.sin(cx * 127.1 + cy * 311.7) * 43758.5453
    h2 = jnp.sin(cx * 269.5 + cy * 183.3) * 43758.5453
    return h1 - jnp.floor(h1), h2 - jnp.floor(h2)

def voronoi(x, y, freq_raw, jitter_raw):
    freq = 1.0 + (freq_raw / 1023) * 8.0
    jitter = jitter_raw / 1023.0
    xs, ys = x * freq, y * freq
    cx0, cy0 = jnp.floor(xs), jnp.floor(ys)
    min_d = jnp.full_like(x, 1e9)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx, cy = cx0 + dx, cy0 + dy
            rx, ry = _cell_hash(cx, cy)
            px = cx + 0.5 + jitter * (rx - 0.5)
            py = cy + 0.5 + jitter * (ry - 0.5)
            min_d = jnp.minimum(min_d, (xs - px) ** 2 + (ys - py) ** 2)
    return jnp.sqrt(min_d) * 2.0 - 1.0

# --- Organic Field (fractal noise, no periodicity/thresholding) ---
def organic_field(x, y, freq_raw, detail_raw):
    freq = 0.15 + (freq_raw / 1023) * 1.5
    persistence = 0.05 + (detail_raw / 1023) * 0.9   # low = smooth/soft, high = rough/detailed
    n = noise.noise2(x * freq, y * freq, octaves=5, persistence=persistence, lacunarity=2.0, grid_mode=False)
    return jnp.clip(n * 1.6, -1.0, 1.0)


input_functions = dict(x=lambda x, y, p1, p2: x,
                       y=lambda x, y, p1, p2: y,
                       symmetry=lambda x, y, p1, p2: rotated_abs(x, y, rot_deg_raw=p1, gain_raw=p2),
                       radial=lambda x, y, p1, p2: generalized_radial(x, y, freq_raw=p1, width_raw=p2),
                       angular=lambda x, y, p1, p2: angular_sinusoid(x, y, phase_raw=p1, amplitude_raw=p2),
                       grid=lambda x, y, p1, p2: grid(x, y, freq_raw=p1, alpha_raw=p2),
                       spiral=lambda x, y, p1, p2: spiral_symmetry(x, y, n_folds_raw=p1, spiral_raw=p2),
                       unif=lambda x, y, p1, p2: jnp.full_like(x, ((p1 / 1023.0) - 0.5) * 6),
                       waves=lambda x, y, p1, p2: waves(x, y, angle_raw=p1, freq_raw=p2),
                       warp=lambda x, y, p1, p2: domain_warp(x, y, freq_raw=p1, strength_raw=p2),
                       voronoi=lambda x, y, p1, p2: voronoi(x, y, freq_raw=p1, jitter_raw=p2),
                       organic=lambda x, y, p1, p2: organic_field(x, y, freq_raw=p1, detail_raw=p2))

# Last selectable slot on the hardware (the one that used to land on 'unif') — currently active:
# 'organic'. 'waves', 'warp', 'voronoi' are implemented above but not wired in; to try one, swap
# the last entry below.
input_selector_mapping = ['x', 'y', 'symmetry', 'radial', 'angular', 'grid', 'spiral', 'organic']


# =====================
# Color space
# =====================
# OKLab is a perceptually-uniform color space: L = lightness, a = green-red, b = blue-yellow
# (Cartesian, like CIELAB). OKLCH is just its polar form (C = chroma, H = hue angle) — same
# relationship HSL has to RGB. Polar/hue-based mapping is why 'oklch' tends to rainbow: hue is an
# angle, so any modest swing in the channel driving it sweeps across the whole color wheel. OKLab's
# a/b are Cartesian offsets with no wraparound, so smooth network output gives smooth, coherent
# color drift instead.

def lin_to_srgb(c):
    """Linear light -> gamma-encoded sRGB, elementwise. Input should be >= 0 (clip first if not —
    a negative base with a fractional exponent below would otherwise NaN)."""
    c = jnp.clip(c, 0.0, None)
    return jnp.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def oklab_to_srgb(L, a, b):
    """OKLab -> gamma-corrected sRGB. L, a, b and outputs are same-shape arrays.
    Not clipped to [0,1] here — some (L,a,b) combos fall outside the sRGB gamut, clip after calling."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    r_lin =  4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g_lin = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_lin = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    return lin_to_srgb(r_lin), lin_to_srgb(g_lin), lin_to_srgb(b_lin)


def oklch_to_srgb(L, C, H):
    """OKLCH (polar form: C=chroma, H=hue in radians) -> gamma-corrected sRGB."""
    return oklab_to_srgb(L, C * jnp.cos(H), C * jnp.sin(H))


# --- Rotated RGB primaries ---
# Stays a plain additive/linear space (like RGB): output = M @ (c0,c1,c2), c_i in [0,1].
# M is built by rotating the standard R,G,B basis around the grey diagonal (1,1,1)/sqrt(3) by
# `hue_deg`, and pulling it `saturation` of the way toward that diagonal. Because the rotation axis
# IS the grey diagonal, M @ (1,1,1) = (1,1,1) exactly for ANY hue_deg/saturation — white and black
# are invariant by construction, only the 3 "primary" colors (the hues you get from a pure channel)
# change. saturation=1 reproduces plain RGB rotated in hue; lower values soften/pastel the primaries.
def rgb_rotation_matrix(hue_deg, saturation):
    theta = jnp.deg2rad(hue_deg)
    n = jnp.ones(3) / jnp.sqrt(3.0)
    I = jnp.eye(3)
    nnT = jnp.outer(n, n)
    n_cross = jnp.array([
        [0.0, -n[2], n[1]],
        [n[2], 0.0, -n[0]],
        [-n[1], n[0], 0.0],
    ])
    return nnT + saturation * (jnp.cos(theta) * (I - nnT) + jnp.sin(theta) * n_cross)


# param mapping

# Mapping functions
def zoom_mapping(value, vmin=0, vmax=1023, log_range=2.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    return 10 ** (norm * log_range)

def bias_mapping(value, vmin=0, vmax=1023, range_=10.0):
    norm = (value - (vmin + vmax) / 2) / ((vmax - vmin) / 2)
    scaled = jnp.sign(norm) * (jnp.exp(jnp.abs(norm) * 3) - 1) / (jnp.exp(3) - 1)
    return range_ * scaled

def inverse_bias_mapping(y, vmin=0, vmax=1023, range_=10.0):
    """Inverse of bias_mapping: raw [vmin,vmax] value that maps to world-space shift y."""
    scaled = y / range_
    norm = jnp.sign(scaled) * jnp.log1p(jnp.abs(scaled) * (jnp.exp(3.0) - 1)) / 3.0
    return (vmin + vmax) / 2 + norm * ((vmax - vmin) / 2)

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