import numpy as np

pull_down = 47000
pull_up = 10000
# resistances = np.array([1000, 1500, 2000, 3000, 4700, 5600, 7500, 8200, 10000, 15000, 22000, 33000, 47000, 68000, 100000])
resistances = np.array([1000, 1500, 2000, 3000, 4700, 5600, 7500, 10000, 15000, 22000, 33000, 47000, 68000, 150000, 680000])

r_eqs = 1 / (1 / resistances + 1/ pull_down)
voltages = r_eqs / (r_eqs + pull_up) * 5
voltages = voltages * (47 / 48)  # accounting for the voltage dividers between the 47 pull down on load side and the 1k safety
steps = (voltages / 5 * 1023).astype(int)
steps_diffs = np.diff(np.concatenate([np.zeros(1), steps]))
thresholds = (steps - steps_diffs // 2).astype(int)
voltage_diffs = np.diff(np.concatenate([np.zeros(1), voltages]))

for i in range(len(voltages)):
    print(f'Node {i + 1}: '
          f'res={resistances[i]}, '
          f'V={voltages[i]:.3f}V, '
          f'step={steps[i]}, '
          # f'diff V: {voltage_diffs[i]:.3f}V, '
          f'diff steps: {int(voltage_diffs[i] / 5 * 1024)}, '
          f'thresholds: {thresholds[i]}')

print(''.join([str(val) + ", " for val in thresholds])[:-2])