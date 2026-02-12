# Audio Reactivity

The visuals react to three frequency bands (bass, mid, treble). Each band produces two signals:

- **EMA** — sustained energy, smooth pumping modulations
- **Flux** — transient attacks (drum hits, plucks), fast spiky modulations

## Reactivity Modes

Set `REACTIVITY` in `src/app.py`:

| Mode | Description |
|------|-------------|
| `"time"` | Visuals evolve on their own. No audio. |
| `"audio"` | Visuals driven by audio only. |
| `"time+audio"` | **Recommended.** Some nodes evolve with time, others respond to audio. |
| `"cv"` | Audio from external interface (Focusrite Scarlett / modular synth). |

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
stale streams keep playing, Spotify loses its connection, audio loops. Here's how to fix it:

1. **Kill stale Python streams**

   Open `pavucontrol` → **Playback** tab. If you see ghost "python3.10" or "ALSA plug-in [python3.10]" entries, kill them:
   ```bash
   pactl list sink-inputs    # find the IDs of stale python streams
   pacmd kill-sink-input <ID>
   ```
   Repeat for each stale stream.

2. **Restart Spotify** (or whatever audio source) — it loses its PulseAudio connection after a crash.

3. **Re-route Spotify** in `pavucontrol` → **Playback** tab → set Spotify output to **Visual_Sink**.

4. **Re-run the app** — `python src/app.py`

If that's not enough (no audio devices show up, weird errors):

5. **Restart PulseAudio**
   ```bash
   pulseaudio -k && sleep 1 && pulseaudio --start
   ```
   Then redo steps 1–4. The virtual sink survives the restart but all routing is reset.

6. **Re-create the virtual sink** (only if it disappeared after step 5)
   ```bash
   pactl list short sinks | grep visual_sink   # check first
   pactl load-module module-null-sink sink_name=visual_sink \
     sink_properties=device.description="Visual_Sink"
   ```

### Troubleshooting

- **No signal**: check pavucontrol — app output must be set to Visual_Sink, not speakers
- **Latency feels off**: adjust audio delay (slider 3 on nanoKONTROL)
- **Debug audio**: run `python src/sound_input.py` standalone for a live visualizer

## CV Mode (Focusrite Scarlett)

For routing a modular synth through a Focusrite USB audio interface.

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
3. Set `REACTIVITY = "time+audio"` in `src/app.py`
4. In pavucontrol, set your audio app's output back to **Visual_Sink**

## MIDI Controls (nanoKONTROL2)

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
| play   | Restart application |
| record | Save state + git push |
| cycle  | Cycle symmetry mode |
| set    | Toggle color inversion |