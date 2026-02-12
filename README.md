# Tangible CPPN

Real-time generative visuals driven by a [Compositional Pattern-Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) (CPPN). The network takes pixel coordinates as input and outputs RGB colors, producing abstract organic patterns that can be shaped in real time via physical controls, MIDI, or audio.

Built for the **Tangible Dreams** installation — visitors sculpt abstract patterns by turning knobs on a custom hardware panel.

## How it works

A CPPN is a small neural network (17 nodes) where each hidden node has a distinct activation function (tanh, cosine, Gaussian, Perlin noise, etc.). By varying the weights, biases, connectivity, and activation functions, radically different visual patterns emerge.

```
5 Input nodes (coordinate functions)
  → 9 Hidden nodes (diverse activations, time/audio-reactive)
    → 3 Output nodes (R, G, B)
```

The forward pass is JIT-compiled with JAX and runs on GPU. At 1024x576, it renders in real time.

## Project structure

```
src/
├── app.py                  # Main entry point
├── cppn.py                 # CPPN network definition and forward pass
├── cppn_utils.py           # Activation functions, input functions, parameter mappings
├── viz.py                  # ModernGL/Pygame rendering + MIDI visual controls
├── sound_input.py          # Real-time audio FFT analysis (bass/mid/treble)
├── midi.py                 # MIDI controller interface (Korg nanoKONTROL2)
├── rs485.py                # RS485 hardware controller (physical installation)
├── github_save.py          # Auto-save outputs + git commit
├── jax_perlin.py           # JAX Perlin noise
├── jax_vnoise/             # Voronoi/Perlin noise implementation
├── arduino/                # Firmware for RS485 hardware nodes
├── communication/          # RS485 protocol tests and utilities
├── misc/                   # Error screens, cycle computation, tests
└── save/                   # Audio processing experiments
docs/
└── audio_reactivity.md     # Audio setup and MIDI control reference
outputs/                    # Saved images and network states
├── mit_stata/              # MIT Stata Center exhibition (Aug 2025)
└── paris/                  # Paris sessions (Nov 2025)
```

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for JAX. See [JAX installation](https://github.com/google/jax#installation) for details.

Additional system packages:
```bash
sudo apt install pavucontrol   # for audio routing (optional)
```

### Run

Edit the config at the top of `src/app.py`:

```python
CONTROLLER = 'midi'       # 'midi' or 'rs485'
SCREEN = 'window'         # 'window', 'laptop', or 'external'
REACTIVITY = "audio+time" # see Reactivity Modes below
RES = 1024                # resolution (height); width = RES * 16/9
```

Then:

```bash
python src/app.py
```

### Keyboard shortcuts (in the visual window)

| Key | Action |
|-----|--------|
| `S` | Save current state (image + network pickle) |
| `N` | Print current network connections to terminal |

## Control modes

### MIDI (Korg nanoKONTROL2)

The nanoKONTROL2 provides physical controls for visual effects and audio parameters. Network parameters (weights, biases, activations) are controlled exclusively by the RS485 hardware panel.

#### Visual controls (`src/viz.py`)

These are always active (no SET button needed):

| CC | Physical | Action |
|----|----------|--------|
| 2 | Knob 3 | Grain/noise overlay (0-100%) |
| 3 | Knob 4 | Displacement strength (0-50px, quadratic) |
| 46 | Button | Cycle symmetry mode (6 modes) |
| 60 | Button | Toggle color inversion |
| 41 | Button | Restart application |
| 45 | Button | Save state + git push |

#### Audio controls (`src/viz.py`, active when audio reactivity is on)

See [docs/audio_reactivity.md](docs/audio_reactivity.md) for full setup instructions.

```
         Col 6          Col 7          Col 8
       ┌─────────┐     ┌────────┐     ┌────────┐
Knob   │ CC 21   │     │ CC 22  │     │ CC 23  │
       │Bass gain│     │Mid gain│     │Treble  │
       │         │     │        │     │gain    │
       ├─────────┤     ├────────┤     ├────────┤
Slider │ CC 5    │     │ CC 6   │     │ CC 4   │
       │Bass/Mid │     │Mid/Treb│     │Audio   │
       │split    │     │split   │     │delay   │
       └─────────┘     └────────┘     └────────┘
```

| CC | Parameter | Range |
|----|-----------|-------|
| 0 | Noise floor baseline | -50 to -20 dB |
| 1 | Gate open threshold | 2-20 dB above baseline |
| 4 | Audio delay | 0-600 ms |
| 5 | Bass/mid crossover | 50-400 Hz (log) |
| 6 | Mid/treble crossover | 500-6000 Hz (log) |
| 21 | Bass gain | 0-6x (quadratic) |
| 22 | Mid gain | 0-6x (quadratic) |
| 23 | Treble gain | 0-6x (quadratic) |

### RS485 hardware (physical installation)

The custom hardware panel connects 17 physical nodes via RS485 serial. Each node has analog potentiometers and switches:

- **Input nodes (1-5):** Select coordinate function, zoom, pan, on/off, invert
- **Middle nodes (6-14):** Select source nodes, set 3 weights, slope, activation function, on/off, CV override
- **Output nodes (15-17):** Select source nodes, set 3 weights, balance, contrast, on/off, CV override

See `src/rs485.py` for the full protocol and sensor mapping.

## Reactivity modes

Set `REACTIVITY` in `src/app.py`:

| Mode | Description |
|------|-------------|
| `"time"` | All hidden nodes oscillate with time-based functions (sine, triangle, Perlin drift). No audio input. |
| `"audio"` | Nodes 4/5/6 respond to treble/mid/bass. Others are static. |
| `"audio+time"` | **Recommended.** Nodes 0-3, 7-8 evolve with time; nodes 4-6 respond to audio. |
| `"cv"` | For RS485 hardware with CV override enabled on nodes. |

## Network architecture

### Input functions

Each of the 5 input nodes selects one coordinate function:

| ID | Function | Param 1 | Param 2 |
|----|----------|---------|---------|
| 0 | `x` | — | — |
| 1 | `y` | — | — |
| 2 | `symmetry` | Rotation angle | Gain |
| 3 | `radial` | Frequency | Width |
| 4 | `angular` | Phase | Amplitude |
| 5 | `grid` | Frequency | Modulation |
| 6 | `spiral` | N-fold symmetry | Spiral tightness |
| 7 | `unif` | Constant value | — |

### Activation functions

Each hidden node can use one of:

| Name | Function |
|------|----------|
| tanh | `tanh(x)` |
| cos | `cos(3x)` |
| leaky_relu | Clipped leaky ReLU |
| gaussian | `exp(-(2x)^2)` |
| modulo | `(x mod 1) * 2 - 1` |
| riemann | Sum of scaled sines (Riemann-zeta inspired) |
| perlin | 1D Perlin noise |
| mysterious | Iterated logistic map `4x(1-x)` |

### Parameter mappings

All analog inputs (0-1023) are mapped through nonlinear functions centered at the midpoint:

| Parameter | Center value | Range | Scaling |
|-----------|-------------|-------|---------|
| Weight | 1.0 | -5 to +5 | Exponential |
| Slope | 1.0 | 0 to 5 | Exponential |
| Zoom | 1.0 | 0.01x to 100x | Logarithmic |
| Bias | 0.0 | -10 to +10 | Exponential |
| Balance | 0.0 | -5 to +5 | Exponential |
| Contrast | 1.0 | 0 to 5 | Exponential |

## Saving outputs

Press Record (MIDI CC 95) or `S` on the keyboard. This saves:
- `outputs/{exp_id}/image_YYYY_MM_DD_HHMMSS.png` — rendered image
- `outputs/{exp_id}/state_YYYY_MM_DD_HHMMSS.pkl` — full network state (loadable)
- Auto-generates a gallery `README.md` in the output directory
- Commits and pushes to git

## Exhibitions

- **MIT Stata Center** (August 2025) — 75+ saved outputs from visitor interactions
- **Paris** (November 2025) — development and demo sessions