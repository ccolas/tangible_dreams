import asyncio
import matplotlib.pyplot as plt
import os
from midi import MIDIController


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


async def main():
    midi = MIDIController(output_path)
    plt.ion()
    factor = 16/9
    # Create figure with initial size but resizable
    fig, ax = plt.subplots(figsize=(6*factor, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    # Generate initial image
    out = midi.cppn.update()
    ax.imshow(out)
    plt.pause(0.016)

    while True:
        try:
            if True: #midi.cppn.needs_update:
                out = midi.cppn.update()
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(out)
                plt.pause(0.00016)
            await asyncio.sleep(0.00016)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    asyncio.run(main())