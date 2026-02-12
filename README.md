# Tangible CPPN

Real-time generative visuals driven by a [Compositional Pattern-Producing Network](https://en.wikipedia.org/wiki/Compositional_pattern-producing_network) (CPPN). The network takes pixel coordinates as input and outputs RGB colors, producing abstract organic patterns that can be shaped in real time via physical controls, MIDI, or audio.

Built for the **Tangible Dreams** installation — visitors sculpt abstract patterns by turning knobs on a custom hardware panel.

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
REACTIVITY = "time+audio"  # see Reactivity Modes below
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
| `"time"` | Visuals evolve on their own. No audio. |
| `"audio"` | Visuals driven by audio only. |
| `"time+audio"` | **Recommended.** Some nodes evolve with time, others respond to audio. |
| `"cv"` | Audio from external interface (Focusrite Scarlett / modular synth). |

See [docs/audio_reactivity.md](docs/audio_reactivity.md) for audio setup, CV mode, and PulseAudio routing.

## MIDI controls (Korg nanoKONTROL2)

The nanoKONTROL2 controls visual effects and audio parameters. Network parameters (weights, biases, activations) are controlled by the RS485 hardware panel.

### Sliders

```
  Slider 1    Slider 2    Slider 3    Slider 4    Slider 5    Slider 6    Slider 7    Slider 8
 ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
 │ Grain  │  │Displace│  │ Audio  │  │  EMA   │  │ Flux   │  │ Bass   │  │  Mid   │  │Treble  │
 │strength│  │strength│  │ delay  │  │sensit. │  │sensit. │  │ gain   │  │ gain   │  │ gain   │
 └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘
   visual      visual      audio       audio       audio       audio       audio       audio
```

| Slider | Parameter | What it does |
|--------|-----------|--------------|
| 1 | Grain | Adds noise/grain overlay |
| 2 | Displacement | Pixel displacement distortion |
| 3 | Audio delay | Delays audio playback to sync with visuals (0-600 ms) |
| 4 | EMA sensitivity | How much sustained energy affects visuals |
| 5 | Flux sensitivity | How much transient attacks affect visuals |
| 6 | Bass gain | Boost/cut bass band reactivity |
| 7 | Mid gain | Boost/cut mid band reactivity |
| 8 | Treble gain | Boost/cut treble band reactivity |

### Buttons

| Button | Action |
|--------|--------|
| Mute 2 | Restart application |
| Mute 6 | Save state + git push |
| Solo 7 | Cycle symmetry mode |
| Record 5 | Toggle color inversion |

## RS485 hardware (physical installation)

The custom hardware panel connects 17 physical nodes via RS485 serial. Each node has analog potentiometers and switches:

- **Input nodes (1-5):** Select coordinate function, zoom, pan, on/off, invert
- **Middle nodes (6-14):** Select source nodes, set 3 weights, slope, activation function, on/off, CV override
- **Output nodes (15-17):** Select source nodes, set 3 weights, balance, contrast, on/off, CV override

## Saving outputs

Press Save (MIDI Mute 6) or `S` on the keyboard. This saves:
- `outputs/{exp_id}/image_YYYY_MM_DD_HHMMSS.png` — rendered image
- `outputs/{exp_id}/state_YYYY_MM_DD_HHMMSS.pkl` — full network state (loadable)
- Commits and pushes to git

## Exhibitions

- **MIT Stata Center** (August 2025) — 75+ saved outputs from visitor interactions
- **Paris** (November 2025) — development and demo sessions