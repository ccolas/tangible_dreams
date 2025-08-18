import asyncio
import os
import time

from src.midi import MIDIController
from src.rs485 import RS485Controller
from src.viz import create_backend

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

exp_id = "mit_stata"
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

async def main():
    params = dict(debug=DEBUG, res=RES, factor=FACTOR)#, load_from="/mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/tangible_cppn/outputs/test//state_2025_06_09_120132.pkl")
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
        width=int(RES * FACTOR),
        height=RES,
        window_scale=1  # Adjust this to make window bigger/smaller
    )

    try:
        # Generate initial image
        # out = controller.cppn.update()
        # viz.update(out)
        while True:
            if controller.cppn.needs_update:
                # t_start = time.time()
                out = controller.cppn.update()
                # t_forward = time.time()
                viz.update(out)
                # t_viz = time.time()
                # time_forward = (t_forward - t_start)  * 1000
                # time_viz = (t_viz - t_forward) * 1000
                # print(f"Times: forward={time_forward:.2f}ms, viz={time_viz:.2f}ms")
            await asyncio.sleep(0.00016)

    except KeyboardInterrupt:
        vis.cleanup()


if __name__ == "__main__":
    asyncio.run(main())