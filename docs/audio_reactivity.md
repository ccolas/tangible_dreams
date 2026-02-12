# Audio Reactivity

The visuals react to three frequency bands (bass, mid, treble). Two audio modes are available:

- **Simple** (`AUDIO_MODE = "simple"`) — 3 signals: per-band EMA (smoothed sustained energy)
- **Flux** (`AUDIO_MODE = "flux"`) — 6 signals: per-band EMA + per-band spectral flux (transient attacks)

## Reactivity Modes

Set `REACTIVITY` in `src/app.py`:

| Mode | Description |
|------|-------------|
| `"time"` | All 9 hidden nodes + outputs evolve with distinct time schedules. No audio. |
| `"audio"` | 3 time nodes + 6 audio-driven nodes. **Recommended.** |
| `"cv"` | 3 time nodes + CV-driven nodes via RS485 hardware. |

Time nodes (5, 6, 13) and output nodes (14-16) are **always** time-controlled in all modes.

## Node-to-signal mapping

| Node (idx) | Time | Audio simple | Audio flux |
|---|---|---|---|
| 5 (0) | sine + harmonics | sine + harmonics | sine + harmonics |
| 6 (1) | Perlin-modulated sine | Perlin-modulated sine | Perlin-modulated sine |
| 7 (2) | slow triangle | bass | bass EMA |
| 8 (3) | slow sine | mid | mid EMA |
| 9 (4) | soft square | bass | bass flux |
| 10 (5) | beating freqs | treble | treble flux |
| 11 (6) | Perlin drift | treble | treble EMA |
| 12 (7) | stepped Perlin | mid | mid flux |
| 13 (8) | Perlin + envelope | Perlin + envelope | Perlin + envelope |
| 14-16 (out) | sine | sine | sine |

In **simple mode**, each band drives 2 nodes (bass: 7+9, mid: 8+12, treble: 10+11).
In **flux mode**, EMA and flux drive separate nodes (EMA: 7,8,11; flux: 9,12,10).

## Audio Setup (PulseAudio)

### 1. Create a virtual sink

```bash
pactl list short sinks | grep visual_sink          # check if it exists
pactl load-module module-null-sink sink_name=visual_sink \
  sink_properties=device.description="Visual_Sink"  # create if not
```

### 2. Route audio to the virtual sink

In `pavucontrol`:

1. **Playback** tab — set your audio app's output to **Visual_Sink**
2. **Output Devices** tab — verify Visual_Sink VU meter is moving

### 3. Run

```bash
python src/app.py
```

### Recovering after a crash

When the app is killed (Ctrl+C, crash, `kill`), PulseAudio can get into a bad state:
stale streams keep playing, Spotify loses its connection, audio loops.

**Quick fix:** run `bash scripts/reset_audio.sh`

**Manual steps:**

1. **Kill stale Python streams**

   Open `pavucontrol` -> **Playback** tab. If you see ghost "python3.10" or "ALSA plug-in [python3.10]" entries, kill them:
   ```bash
   pactl list sink-inputs    # find the IDs of stale python streams
   pacmd kill-sink-input <ID>
   ```
   Repeat for each stale stream.

2. **Restart Spotify** (or whatever audio source) — it loses its PulseAudio connection after a crash.

3. **Re-route Spotify** in `pavucontrol` -> **Playback** tab -> set Spotify output to **Visual_Sink**.

4. **Re-run the app** — `python src/app.py`

If that's not enough (no audio devices show up, weird errors):

5. **Restart PulseAudio**
   ```bash
   pulseaudio -k && sleep 1 && pulseaudio --start
   ```
   Then redo steps 1-4. The virtual sink survives the restart but all routing is reset.

6. **Re-create the virtual sink** (only if it disappeared after step 5)
   ```bash
   pactl list short sinks | grep visual_sink   # check first
   pactl load-module module-null-sink sink_name=visual_sink \
     sink_properties=device.description="Visual_Sink"
   ```

### Troubleshooting

- **No signal**: check pavucontrol — app output must be set to Visual_Sink, not speakers
- **Latency feels off**: adjust audio delay slider
- **Debug audio**: run `python src/sound_input.py` standalone for a live visualizer

## CV Mode (Focusrite Scarlett)

For routing a modular synth through a Focusrite USB audio interface. CV modulation goes through RS485 hardware (not the Scarlett).

### Setup

1. Plug Scarlett into USB. Patch synth into **Input 1**. Gain at ~50% (green halo = good, red = clipping).
2. Verify it appears in `pavucontrol` -> **Input Devices** with a moving VU meter.
3. Create a loopback to bridge the input to the visual sink:

   ```bash
   pactl load-module module-loopback latency_msec=30
   ```

4. In `pavucontrol`:
   - **Recording** tab — set loopback input to **Scarlett**
   - **Playback** tab — set loopback output to **Visual_Sink**

5. Set `REACTIVITY = "cv"` in `src/app.py` and run.

### Managing loopbacks

```bash
pactl list short modules | grep loopback   # list (note the index number)
pactl unload-module <index>                # remove one
pactl unload-module module-loopback        # remove all
```

Loopbacks don't persist across reboots.

### Switching back to audio mode

The loopback conflicts with normal audio mode. **Remove it first**, then switch back:

1. `pactl list short modules | grep loopback` — find the index
2. `pactl unload-module <index>` — remove it
3. Set `REACTIVITY = "audio"` in `src/app.py`
4. In pavucontrol, set your audio app's output back to **Visual_Sink**

## MIDI Controls (nanoKONTROL2)

### Sliders

Two layouts depending on `AUDIO_MODE`:

**Flux mode:**

| Slider | CC | Parameter | Range |
|--------|----|-----------|-------|
| 1 | 0 | Grain strength | 0-0.7 |
| 2 | 1 | Displacement | 0-3% screen width |
| 3 | 2 | Audio delay | 0-150 ms |
| 4 | 3 | EMA attack speed | 0-0.95 |
| 5 | 4 | EMA release speed | 0-0.95 |
| 6 | 5 | Bass gain | 0-8x |
| 7 | 6 | Mid gain | 0-8x |
| 8 | 7 | Treble gain | 0-8x |

**Simple mode:**

| Slider | CC | Parameter | Range |
|--------|----|-----------|-------|
| 1 | 0 | Grain strength | 0-0.7 |
| 2 | 1 | Displacement | 0-3% screen width |
| 3 | 2 | EMA attack speed | 0-0.95 |
| 4 | 3 | EMA release speed | 0-0.95 |
| 5 | 4 | Audio delay | 0-150 ms |
| 6 | 5 | Bass gain | 0-8x |
| 7 | 6 | Mid gain | 0-8x |
| 8 | 7 | Treble gain | 0-8x |

### Knobs

| Knob | CC | Parameter | Range |
|------|----|-----------|-------|
| 2 | 17 | Flux decay (flux mode only) | 0.50-0.99 |
| 3 | 18 | Bass low cutoff | 20-100 Hz |
| 4 | 19 | Bass/mid crossover | 50-500 Hz |
| 5 | 20 | Mid/treble crossover | 1000-6000 Hz |
| 6 | 21 | Bass gate open % | 0-40% |
| 7 | 22 | Mid gate open % | 0-40% |
| 8 | 23 | Treble gate open % | 0-40% |

### Buttons

| Button | CC | Action |
|--------|----|--------|
| play | 41 | Restart application |
| record | 45 | Save state + git push |
| cycle | 46 | Cycle symmetry mode |
| set | 60 | Toggle color inversion |
| R3 | 66 | Set delay to measured render time (flux mode) |
| R5 | 68 | Set delay to measured render time (simple mode) |
