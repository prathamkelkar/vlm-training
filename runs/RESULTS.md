# Pilot results (test split)

Accuracy on Ego-EXTRA multiple-choice questions. Chance = 20.0%. ± is a 95% interval (normal approx.).

| Arm | With video | Blind (no video) | Video gain | Predicted letters (video) | Letter mass |
|---|---|---|---|---|---|
| arm_b_seed0 | 70.7 ± 2.2 | 67.6 ± 2.2 | +3.1 | {'A': 316, 'B': 351, 'C': 364, 'D': 322, 'E': 344} | 1.00 |
| arm_c_seed0 | 68.7 ± 2.2 | 64.6 ± 2.3 | +4.0 | {'A': 342, 'B': 359, 'C': 336, 'D': 321, 'E': 339} | 1.00 |
| zero_shot | 41.4 ± 2.3 | 36.5 ± 2.3 | +4.9 | {'A': 354, 'B': 368, 'C': 460, 'D': 257, 'E': 258} | 1.00 |

## By scenario (with video)

| Arm | 01 | 02 | 03 | 04 |
|---|---|---|---|---|
| arm_b_seed0 | 72.7 | 64.8 | 67.0 | 76.3 |
| arm_c_seed0 | 71.6 | 62.5 | 65.5 | 73.0 |
| zero_shot | 40.5 | 32.0 | 41.7 | 43.6 |

## Training runs

| Run | Best val acc | Optimizer steps | Best adapter |
|---|---|---|---|
| arm_b_seed0 | 69.3 | 626 | runs/arm_b_seed0/best |
| arm_c_seed0 | 69.5 | 780 | runs/arm_c_seed0/best |
