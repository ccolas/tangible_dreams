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

# TODO
# calibrate each activation function
# calibrate default zoom size
# calibrate default slopes of output nodes
# calibrate initial probability of connection
# select carefully activation functions
# optimize for speed with claude
# when saving, save with higher res
# add button to save as elites, then mutate these when sampling new network!

DEBUG = False

async def main():
    midi = MIDIController(output_path, debug=DEBUG)
    plt.ion()
    factor = 16/9
    size = 1500

    vis = create_backend('moderngl')  # or 'opencv'
    vis.initialize(int(size * factor), size)

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