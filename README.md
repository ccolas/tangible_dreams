# Tangible Dreams

Real-time generative visuals driven by a [Compositional Pattern-Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) (CPPN). The network takes pixel coordinates as input and outputs RGB colors, producing abstract organic patterns that can be shaped in real time via physical controls, MIDI, or audio.

Built for the **[Tangible Dreams](https://cedriccolas.com/project/tangible-dreams)** installation — visitors sculpt abstract patterns by turning knobs on a custom hardware panel.

## How it works

A CPPN is a small neural network (17 nodes) where each hidden node has a distinct activation function (tanh, cosine, Gaussian, Perlin noise, etc.). By varying the weights, biases, connectivity, and activation functions, radically different visual patterns emerge.

```
5 Input nodes (coordinate functions)
  -> 9 Hidden nodes (diverse activations, time/audio-reactive)
    -> 3 Output nodes (R, G, B)
```

## Quick start

```bash
pip install -r requirements.txt          # requires CUDA-capable GPU for JAX
sudo apt install pavucontrol             # for audio routing (optional)
```

Edit the config at the top of `src/app.py`:

```python
CONTROLLER = 'midi'        # 'midi' or 'rs485'
SCREEN = 'window'          # 'window', 'laptop', or 'external'
REACTIVITY = "audio"       # 'time', 'audio', or 'cv'
AUDIO_MODE = "simple"      # 'simple' (3 EMA signals) or 'flux' (3 EMA + 3 flux)
RES = 1024                 # resolution (height); width = RES * 16/9
```

```bash
python src/app.py
```

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| `S` | Save current state (image + network pickle) |
| `N` | Print current network connections to terminal |

## Reactivity modes

Set `REACTIVITY` in `src/app.py`:

| Mode | Description |
|------|-------------|
| `"time"` | All nodes evolve with time using distinct schedules. No audio. |
| `"audio"` | 3 time nodes + 6 audio-driven nodes (EMA/flux). **Recommended.** |
| `"cv"` | 3 time nodes + CV-driven nodes via RS485 hardware. |

In all modes, **the top-row nodes (6, 11, 14) and output nodes (15-17) are always active** with distinct time-based dynamics (perlin, sine, beating frequencies).

## Hardware node layout

The 17 physical nodes are arranged on the panel in 5 columns (top to bottom):

```
        col1  col2  col3  col4  col5
row1  [  5  ][  6  ][ 11  ][ 14  ][ 16  ]
row2  [  4  ][  7  ][ 10  ][ 12  ][ 17  ]
row3  [  3  ][  8  ][  9  ][ 13  ][ 15  ]
row4  [  2  ][  1  ][     ][     ][     ]
```

- Inputs (1-5) span col1 (5,4,3,2 top to bottom) plus node 1 at the bottom of col2.
- Hidden nodes (6-14) fill the top 3 rows of col2-col4.
- Outputs (15-17) fill the top 3 rows of col5.
- Top row (6, 11, 14) is always time-driven, regardless of reactivity mode.

## Audio modes

Set `AUDIO_MODE` in `src/app.py`:

| Mode | Signals | Viz | Description |
|------|---------|-----|-------------|
| `"simple"` | 3 (low/mid/high EMA) | 2x2 | Each band has 2 nodes sharing the same EMA signal. |
| `"flux"` | 6 (3 EMA + 3 flux) | 2x3 | Separate EMA (sustained) and flux (transient) per band. |

### Node-to-signal mapping

| Node | Time (fallback) | Audio simple | Audio flux |
|---|---|---|---|
| 6 | perlin drift | perlin drift (always) | perlin drift (always) |
| 7 | Perlin-modulated sine | low EMA | low EMA |
| 8 | slow triangle | low EMA | low flux |
| 9 | phase-shifted sine | mid EMA | mid EMA |
| 10 | soft square | mid EMA | mid flux |
| 11 | plain sine | sine (always) | sine (always) |
| 12 | sine + harmonics | high EMA | high EMA |
| 13 | stepped Perlin | high EMA | high flux |
| 14 | beating freqs | beating freqs (always) | beating freqs (always) |
| 15-17 (out) | sine | sine | sine |

Nodes 6, 11, 14 (top row) are **always** perlin / sine / beating, in every reactivity mode. Nodes 7-10 and 12-13 drive low/mid/high in `"audio"` mode, and fall back to their own distinct time schedule in `"time"` mode.

See [docs/audio_reactivity.md](docs/audio_reactivity.md) for audio setup, MIDI controls, and PulseAudio routing.

## MIDI controls (Korg nanoKONTROL2)

The nanoKONTROL2 controls visual effects and audio parameters. Network parameters (weights, biases, activations) are controlled by the RS485 hardware panel.

### Sliders

Two layouts depending on `AUDIO_MODE`:

**Flux mode** (CC mapping):

```
  Slider 1    Slider 2    Slider 3    Slider 4    Slider 5    Slider 6    Slider 7    Slider 8
 CC 0        CC 1        CC 2        CC 3        CC 4        CC 5        CC 6        CC 7
 ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
 │ Grain  │  │Displace│  │ Audio  │  │ Attack │  │Release │  │ Bass   │  │  Mid   │  │Treble  │
 │strength│  │strength│  │ delay  │  │ speed  │  │ speed  │  │ gain   │  │ gain   │  │ gain   │
 └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
```

**Simple mode** (CC mapping):

```
  Slider 1    Slider 2    Slider 3    Slider 4    Slider 5    Slider 6    Slider 7    Slider 8
 CC 0        CC 1        CC 2        CC 3        CC 4        CC 5        CC 6        CC 7
 ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
 │ Grain  │  │Displace│  │ Attack │  │Release │  │ Audio  │  │ Bass   │  │  Mid   │  │Treble  │
 │strength│  │strength│  │ speed  │  │ speed  │  │ delay  │  │ gain   │  │ gain   │  │ gain   │
 └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
```

### Knobs

| Knob (CC) | Parameter | Range |
|-----------|-----------|-------|
| 17 | Flux decay (flux mode only) | 0.50-0.99 |
| 18 | Bass low cutoff | 20-100 Hz |
| 19 | Bass/mid crossover | 50-500 Hz |
| 20 | Mid/treble crossover | 1000-6000 Hz |
| 21 | Bass gate open % | 0-40% |
| 22 | Mid gate open % | 0-40% |
| 23 | Treble gate open % | 0-40% |

### Buttons

| Button (CC) | Action |
|-------------|--------|
| 41 | Restart application |
| 45 | Save state + git push |
| 46 | Cycle symmetry mode |
| 60 | Toggle color inversion |
| 66 | Set delay to measured render time (flux mode) |
| 68 | Set delay to measured render time (simple mode) |

## RS485 hardware (physical installation)

The custom hardware panel connects 17 physical nodes via RS485 serial. Each node has analog potentiometers and switches:

- **Input nodes (1-5):** Select coordinate function, zoom, pan, on/off, invert
- **Middle nodes (6-14):** Select source nodes, set 3 weights, slope, activation function, on/off, CV override
- **Output nodes (15-17):** Select source nodes, set 3 weights, balance, contrast, on/off, CV override

## Saving outputs

Press Save (MIDI CC 45) or `S` on the keyboard. This saves:
- `outputs/{exp_id}/image_YYYY_MM_DD_HHMMSS.png` — rendered image
- `outputs/{exp_id}/state_YYYY_MM_DD_HHMMSS.pkl` — full network state (loadable)
- Commits and pushes to git

## Exhibitions

- **MIT Stata Center** (August 2025) — 75+ saved outputs from visitor interactions
- **Paris** (November 2025) — development and demo sessions
