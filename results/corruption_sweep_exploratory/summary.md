# Corruption sweep, exploratory behaviour figures


Outcome v2 = frozen oracle (confirmatory). Destination-aware = v2, except failures whose manipulated object headed for another task's destination (protocol oracle.destination_label).

| condition | n | success | wrong task (v2) | right task, botched (v2) | of which wrong destination | stalled | collision | off table |
|---|---|---|---|---|---|---|---|---|
| identity | 300 | 292 | 1 | 6 | 0 (all: 0) | 1 | 3 | 0 |
| bg_patches-0.8 | 300 | 254 | 4 | 42 | 1 (all: 1) | 0 | 6 | 0 |
| blur-1 | 300 | 295 | 1 | 4 | 0 (all: 0) | 0 | 5 | 0 |
| blur-4 | 300 | 240 | 2 | 55 | 3 (all: 4) | 3 | 2 | 0 |
| blur-8 | 300 | 73 | 25 | 121 | 16 (all: 17) | 81 | 32 | 0 |
| jpeg-6 | 300 | 290 | 1 | 9 | 0 (all: 0) | 0 | 4 | 0 |
| jpeg-1 | 300 | 147 | 8 | 124 | 0 (all: 3) | 21 | 4 | 0 |
| matched_bg_box | 300 | 290 | 2 | 7 | 3 (all: 3) | 1 | 6 | 0 |
| noise-64 | 300 | 269 | 0 | 30 | 0 (all: 0) | 1 | 8 | 0 |
| noise-128 | 300 | 95 | 33 | 101 | 16 (all: 23) | 71 | 7 | 0 |
| photometric-3.5 | 300 | 277 | 0 | 23 | 1 (all: 1) | 0 | 1 | 0 |
| photometric-4.5 | 300 | 177 | 22 | 74 | 1 (all: 1) | 27 | 10 | 0 |
| target_occlusion | 300 | 42 | 38 | 163 | 9 (all: 10) | 57 | 72 | 0 |
| bg_patches-0.8-exactmi | 300 | 246 | 1 | 52 | 0 (all: 0) | 1 | 10 | 0 |
