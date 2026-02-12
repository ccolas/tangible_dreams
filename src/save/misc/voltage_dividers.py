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


# /mnt/e85692fd-9cbc-4a8d-b5c5-9252bd9a34fd/Perso/Scratch/tangible_cppn/src/misc/voltage_dividers.py
# Node  1: res=   510, V_1jack=0.242V, V_3jack=0.241V, step_1jack=49, step_3jack=49 | margin from prev: 49 | threshold: 24
# Node  2: res=  1000, V_1jack=0.453V, V_3jack=0.449V, step_1jack=92, step_3jack=91 | margin from prev: 42 | threshold: 70
# Node  3: res=  1500, V_1jack=0.649V, V_3jack=0.643V, step_1jack=132, step_3jack=131 | margin from prev: 39 | threshold: 111
# Node  4: res=  2000, V_1jack=0.829V, V_3jack=0.820V, step_1jack=169, step_3jack=167 | margin from prev: 35 | threshold: 149
# Node  5: res=  3000, V_1jack=1.146V, V_3jack=1.132V, step_1jack=234, step_3jack=231 | margin from prev: 62 | threshold: 200
# Node  6: res=  4700, V_1jack=1.586V, V_3jack=1.560V, step_1jack=324, step_3jack=319 | margin from prev: 85 | threshold: 276
# Node  7: res=  5600, V_1jack=1.779V, V_3jack=1.747V, step_1jack=363, step_3jack=357 | margin from prev: 33 | threshold: 340
# Node  8: res=  7500, V_1jack=2.121V, V_3jack=2.078V, step_1jack=433, step_3jack=425 | margin from prev: 62 | threshold: 394
# Node  9: res= 10000, V_1jack=2.471V, V_3jack=2.414V, step_1jack=505, step_3jack=493 | margin from prev: 60 | threshold: 463
# Node 10: res= 15000, V_1jack=2.959V, V_3jack=2.881V, step_1jack=605, step_3jack=589 | margin from prev: 84 | threshold: 547
# Node 11: res= 22000, V_1jack=3.385V, V_3jack=3.285V, step_1jack=692, step_3jack=672 | margin from prev: 67 | threshold: 638
# Node 12: res= 33000, V_1jack=3.773V, V_3jack=3.650V, step_1jack=771, step_3jack=746 | margin from prev: 54 | threshold: 719
# Node 13: res= 56000, V_1jack=4.165V, V_3jack=4.017V, step_1jack=852, step_3jack=821 | margin from prev: 50 | threshold: 796
# Node 14: res=150000, V_1jack=4.594V, V_3jack=4.417V, step_1jack=939, step_3jack=903 | margin from prev: 51 | threshold: 877
