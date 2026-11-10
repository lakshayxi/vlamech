# Cross-model check: pi0.5 against OFT (held-out, 300 episodes per cell)

| condition | pi0.5 success | OFT success | pi0.5 change (pts) | OFT change (pts) | pi0.5 - OFT (pts) | p_holm (pi0.5) |
|---|---|---|---|---|---|---|
| clean | 98.7% | 97.3% | - | - | - | - |
| blur-8 | 0.0% | 24.3% | -98.7 [-99.7, -97.3] | -73.0 [-77.0, -69.0] | -25.7 [-30.0, -21.7] | 3.14e-89 |
| noise-128 | 1.3% | 31.7% | -97.3 [-99.0, -95.3] | -65.7 [-69.7, -61.7] | -31.7 [-36.0, -27.3] | 2.51e-88 |
