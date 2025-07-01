import numpy as np

pull_down_1 = 510000  # 1 jack connected
pull_down_3 = pull_down_1 / 3  # 3 jacks connected in parallel
pull_up = 10000
r_safety = 1000  # corrected value

resistances = np.array([510,
    1000, 1500, 2000, 3000, 4700, 5600, 7500,
    10000, 15000, 22000, 33000, 56000, 150000
])

# Compute voltages for 1 jack connected
r_eqs_1 = 1 / (1 / resistances + 1 / pull_down_1)
v_divider_1 = r_eqs_1 / (r_eqs_1 + pull_up) * 5
voltages_1 = v_divider_1 * (pull_down_1 / (pull_down_1 + r_safety))

# Compute voltages for 3 jacks connected
r_eqs_3 = 1 / (1 / resistances + 1 / pull_down_3)
v_divider_3 = r_eqs_3 / (r_eqs_3 + pull_up) * 5
voltages_3 = v_divider_3 * (pull_down_3 / (pull_down_3 + r_safety))

# Convert to ADC steps
steps_1 = (voltages_1 / 5 * 1023).astype(int)
steps_3 = (voltages_3 / 5 * 1023).astype(int)

# Compute margins and thresholds between adjacent levels
step_margins = []
thresholds = []

for i in range(len(resistances)):
    if i == 0:
        margin = steps_3[i]
        threshold = steps_3[i] // 2
    else:
        margin = steps_3[i] - steps_1[i - 1]
        threshold = (steps_1[i - 1] + steps_3[i]) // 2
    step_margins.append(margin)
    thresholds.append(threshold)

# Print results
for i in range(len(resistances)):
    margin_str = f" | margin from prev: {step_margins[i]}"
    threshold_str = f" | threshold: {thresholds[i]}"
    print(f'Node {i + 1:2}: '
          f'res={resistances[i]:>6}, '
          f'V_1jack={voltages_1[i]:.3f}V, '
          f'V_3jack={voltages_3[i]:.3f}V, '
          f'step_1jack={steps_1[i]}, '
          f'step_3jack={steps_3[i]}'
          f'{margin_str}{threshold_str}')

# Thresholds summary
print('\nThresholds array:')
print(', '.join(str(val) for val in thresholds))
