import asyncio
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import jax.numpy as jnp
import time

from midi import MIDIController
from viz import create_backend

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

exp_id = "test"
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

DEBUG = False
RES = 2048
FACTOR = 16/9

async def main():
    params = dict(debug=DEBUG, res=RES, factor=FACTOR)
    midi = MIDIController(output_path, params)
    plt.ion()

    vis = create_backend('pygame')  # or 'opencv'
    vis.initialize(
        render_width=int(RES * FACTOR),
        render_height=RES,
        window_scale=0.5  # Adjust this to make window bigger/smaller
    )

    try:
        # Generate initial image
        out = midi.cppn.update()
        vis.update(out)

        while True:
            if DEBUG: midi.check_midi()
            if midi.cppn.needs_update:
                out = midi.cppn.update()
                times = vis.update(out)
                print("Visualization times:", times)
            await asyncio.sleep(0.00016)

    except KeyboardInterrupt:
        vis.cleanup()


if __name__ == "__main__":
    asyncio.run(main())