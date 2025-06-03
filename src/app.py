import asyncio
import os

from src.midi import MIDIController
from src.rs845 import RS485Controller
from src.viz import create_backend

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

DEBUG = True
RES = 1024
FACTOR = 16/9
CONTROLLER = 'midi'

async def main():
    params = dict(debug=DEBUG, res=RES, factor=FACTOR)
    if CONTROLLER == 'midi':
        controller = MIDIController(output_path, params)
        asyncio.create_task(controller.start_polling_loop())  # async MIDI polling
    elif CONTROLLER == 'rs845':
        controller = RS485Controller(output_path, params)
        asyncio.create_task(controller.start_polling_loop())  # run in background
    else:
        raise NotImplementedError

    vis = create_backend('pygame')
    vis.initialize(
        render_width=int(RES * FACTOR),
        render_height=RES,
        window_scale=1  # Adjust this to make window bigger/smaller
    )

    try:
        # Generate initial image
        out = controller.cppn.update()
        vis.update(out)

        while True:
            if controller.cppn.needs_update:
                out = controller.cppn.update()
                times = vis.update(out)
                print("Visualization times:", times)
            await asyncio.sleep(0.00016)

    except KeyboardInterrupt:
        vis.cleanup()


if __name__ == "__main__":
    asyncio.run(main())