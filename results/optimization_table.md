# Phase 4: Compiler Optimization Results

## Execution Time (ms/iter, 100 iterations, 512x512, VLEN=512)

| Stage              | -O0      | -O2     | -O3     | -Os     | -Ofast  |
|--------------------|----------|---------|---------|---------|---------|
| Gaussian spatial   | 1342.077 | 17.135  | 7.372   | 19.518  | 7.478   |
| Gaussian separable | 702.679  | 7.062   | 3.977   | 8.795   | 4.036   |
| Sobel              | 5.755    | 2.475   | 2.442   | 1.969   | 2.546   |
| Magnitude L1       | 161.299  | 7.072   | 6.733   | 7.033   | 6.788   |
| Magnitude L2       | 197.638  | 11.853  | 12.203  | 34.805  | 15.351  |
| Direction          | 59.017   | 3.067   | 3.084   | 3.097   | 3.133   |

## Binary Sizes

| Flag    | Size  |
|---------|-------|
| -O0     | 3.6M  |
| -O2     | 2.3M  |
| -O3     | 2.3M  |
| -Os     | 3.0M  |
| -Ofast  | 2.3M  |

## Key Observations

- Gaussian spatial is the heaviest stage: 1342ms at -O0, drops 182x to 7.4ms at -O3
- Gaussian separable is ~2x faster than spatial at every level — separable filter pays off
- -O3 and -Ofast are nearly identical, meaning floating-point relaxation gives no benefit here
- -Os (optimize for size) is slower than -O3 for most stages, but binary size is 3.0M vs 2.3M — worse on both counts
- Magnitude L2 spikes at -Os (34.8ms vs 12.2ms at -O3) because sqrt() is no longer inlined
- Sobel and Direction are cheap at all levels — not worth vectorizing manually (Amdahl's law)
- Hotspots for Phase 6 RVV optimization: Gaussian spatial (7.4ms) and Magnitude L1 (6.7ms)
