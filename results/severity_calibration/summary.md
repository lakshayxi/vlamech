# Severity calibration: success vs severity

Calibration split, 10 tasks x 5 init states, both cameras. Clean: 50/50.

## photometric (stops darker)

| level | success | 95% CI | isotonic fit / clean |
|---|---|---|---|
| 0.5 | 50/50 | [93%, 100%] | 1.00 |
| 1 | 50/50 | [93%, 100%] | 1.00 |
| 1.5 | 47/50 | [84%, 98%] | 0.98 |
| 2 | 49/50 | [90%, 100%] | 0.98 |
| 2.5 | 50/50 | [93%, 100%] | 0.98 |
| 3 | 50/50 | [93%, 100%] | 0.98 |
| 3.5 | 45/50 | [79%, 96%] | 0.90 |
| 4 | 41/50 | [69%, 90%] | 0.82 |
| 4.5 | 27/50 | [40%, 67%] | 0.54 |

Nearest level per target: 90% -> 3.5, 60% -> 4.5, 30% -> 4.5
Selected levels: [3.5, 4.5]; unreachable targets [0.3]
Ladder extension (rule): 4.5

## noise (noise σ (0-255))

| level | success | 95% CI | isotonic fit / clean |
|---|---|---|---|
| 2 | 49/50 | [90%, 100%] | 0.98 |
| 4 | 48/50 | [87%, 99%] | 0.97 |
| 8 | 48/50 | [87%, 99%] | 0.97 |
| 16 | 50/50 | [93%, 100%] | 0.97 |
| 32 | 49/50 | [90%, 100%] | 0.97 |
| 64 | 48/50 | [87%, 99%] | 0.96 |
| 128 | 15/50 | [19%, 44%] | 0.30 |

Nearest level per target: 90% -> 64, 60% -> 128, 30% -> 128
Selected levels: [64.0, 128.0]

## blur (blur σ (px))

| level | success | 95% CI | isotonic fit / clean |
|---|---|---|---|
| 0.5 | 50/50 | [93%, 100%] | 1.00 |
| 1 | 48/50 | [87%, 99%] | 0.97 |
| 2 | 49/50 | [90%, 100%] | 0.97 |
| 4 | 41/50 | [69%, 90%] | 0.82 |
| 8 | 12/50 | [14%, 37%] | 0.24 |
| 16 | 4/50 | [3%, 19%] | 0.08 |

Nearest level per target: 90% -> 1, 60% -> 4, 30% -> 8
Selected levels: [1.0, 4.0, 8.0]

## jpeg (JPEG quality)

| level | success | 95% CI | isotonic fit / clean |
|---|---|---|---|
| 96 | 50/50 | [93%, 100%] | 1.00 |
| 48 | 47/50 | [84%, 98%] | 0.95 |
| 24 | 48/50 | [87%, 99%] | 0.95 |
| 12 | 48/50 | [87%, 99%] | 0.95 |
| 6 | 43/50 | [74%, 93%] | 0.86 |
| 3 | 38/50 | [63%, 86%] | 0.76 |
| 1 | 27/50 | [40%, 67%] | 0.54 |

Nearest level per target: 90% -> 6, 60% -> 1, 30% -> 1
Selected levels: [6.0, 1.0]; unreachable targets [0.3]
Ladder extension (rule): none possible (physical bound)

## bg_patches (background coverage)

| level | success | 95% CI | isotonic fit / clean |
|---|---|---|---|
| 0.05 | 49/50 | [90%, 100%] | 0.99 |
| 0.1 | 50/50 | [93%, 100%] | 0.99 |
| 0.2 | 49/50 | [90%, 100%] | 0.98 |
| 0.4 | 48/50 | [87%, 99%] | 0.96 |
| 0.8 | 43/50 | [74%, 93%] | 0.86 |

Nearest level per target: 90% -> 0.8, 60% -> 0.8, 30% -> 0.8
Selected levels: [0.8]; unreachable targets [0.3, 0.6]
Ladder extension (rule): none possible (physical bound)

