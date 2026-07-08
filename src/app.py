import asyncio
import os
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MIDIController/RS485Controller/create_backend are imported inside main() (not here) — the video
# recorder's assembly step spawns worker processes via multiprocessing 'spawn', which re-executes
# this file's top-level code (though not main() itself, thanks to the __name__ guard below). Heavy
# imports up here (pygame, moderngl, rtmidi, serial) would otherwise load — and print banners —
# uselessly in every one of those workers.

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

exp_id = "jul_6"
output_path = repo_path + f'outputs/{exp_id}/'
os.makedirs(output_path, exist_ok=True)

# TODO:
#  calibrate each activation function and inputs
#  make it audio reactive?
#  calibrate default zoom size
#  calibrate default slopes of output nodes
#  calibrate initial probability of connection
#  select carefully activation functions
#  optimize for speed with claude
#  add button to save as elites, then mutate these when sampling new network!

DEBUG = True
RES = 1024
FACTOR = 16/9
CONTROLLER = 'rs485'
SCREEN = 'external'  # or 'external' or 'window'
REACTIVITY = "audio"  # "time", "audio", or "cv"
AUDIO_MODE = "simple"  # "simple" (3 EMA signals) or "flux" (3 EMA + 3 flux)
VISUALIZE_AUDIO = True
COLOR_SPACE = "rgb_rotated"  # "rgb" (legacy), "oklch" (hue/chroma/lightness — tends to rainbow), or
                        # "oklab" (lightness/a/b, Cartesian — no hue wraparound, smoother drift)

def update_reactivity(cppn):
    """Update audio/time weight mods. Called once per frame.
    All updates collected in numpy, single JAX transfer at the end."""
    import numpy as np
    from jax import numpy as jnp
    if not any(m in cppn.reactivity for m in ("time", "audio", "cv")):
        return
    cv_override = np.asarray(cppn.device_state['cv_override'])
    weight_mods = np.array(cppn.device_state['weight_mods'])  # copy — np.asarray returns read-only view of JAX array
    changed = False
    for node_idx in range(cppn.n_inputs, cppn.n_nodes):
        if cv_override[node_idx]:
            weight1 = cppn.weights_1_raw[node_idx]
            weight2 = cppn.reactive_update(node_idx, weight1)
            if weight2 is not None:
                source2_idx = cppn.inputs_nodes_record[node_idx, 1]
                if source2_idx >= 0:
                    weight_mods[source2_idx, node_idx] = np.float16(weight2)
                changed = True
    if changed:
        cppn.device_state['weight_mods'] = jnp.array(weight_mods, dtype=jnp.float16)
        cppn.needs_update = True

    # Input nodes: sine-driven shift replaces the manual value while their switch is on
    input_shift_mods = np.array(cppn.device_state['input_shift_mods'])
    input_changed = False
    for node_idx in range(cppn.n_inputs):
        if cv_override[node_idx]:
            new_val = cppn.input_reactive_update(node_idx)
            if new_val != input_shift_mods[node_idx]:
                input_shift_mods[node_idx] = new_val
                input_changed = True
    if input_changed:
        cppn.device_state['input_shift_mods'] = jnp.array(input_shift_mods, dtype=jnp.float32)
        cppn.needs_update = True

async def main():
    from save.midi import MIDIController
    from src.rs485 import RS485Controller
    from src.viz import create_backend

    params = dict(debug=DEBUG, res=RES, factor=FACTOR, visualize_audio=VISUALIZE_AUDIO,
                  reactivity=REACTIVITY, audio_mode=AUDIO_MODE, color_space=COLOR_SPACE)#, load_from="/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/tangible_cppn/outputs/test//state_2025_06_09_120132.pkl")
    if CONTROLLER == 'midi':
        controller = MIDIController(output_path, params)
        asyncio.create_task(controller.start_polling_loop())  # async MIDI polling
    elif CONTROLLER == 'rs485':
        controller = RS485Controller(output_path, params)
        asyncio.create_task(controller.start_polling_loop())  # run in background
    else:
        raise NotImplementedError

    viz = create_backend('moderngl')
    controller.viz = viz
    viz.initialize(
        cppn=controller.cppn,
        screen_loc=SCREEN,
        width=int(RES * FACTOR),
        height=RES,
        window_scale=1  # Adjust this to make window bigger/smaller
    )

    try:
        from collections import deque
        frame_times = deque(maxlen=30)
        last_frame_time = time.time()
        while True:
            t0 = time.time()
            update_reactivity(controller.cppn)
            t_react = time.time()
            if controller.cppn.needs_update:
                out = controller.cppn.update()
                t_forward = time.time()
                viz.update(out)
                t_end = time.time()
                time_react = (t_react - t0) * 1000
                time_forward = (t_forward - t_react) * 1000
                time_viz = (t_end - t_forward) * 1000
                dt = t_end - last_frame_time
                last_frame_time = t_end
                frame_times.append(t_end - t_react)  # render time only
                viz.measured_delay = sum(frame_times) / len(frame_times)
                fps = 1.0 / dt if dt > 0 else 0
                delay_ms = (controller.cppn.audio.delay_seconds * 1000) if controller.cppn.audio else 0
                # print(f"react={time_react:.1f}ms fwd={time_forward:.1f}ms viz={time_viz:.1f}ms fps={fps:.0f} delay={delay_ms:.0f}ms")
            elif viz.needs_update:
                viz.update(out)
            await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        viz.cleanup()


if __name__ == "__main__":
    asyncio.run(main())