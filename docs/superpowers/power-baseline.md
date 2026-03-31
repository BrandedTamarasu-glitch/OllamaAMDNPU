# Task 7: NPU Power Baseline

**Date:** 2026-03-31 05:20:24
**Workload:** single_core i8 matmul 64×2048×64 (tile 64×64×64), 200 iterations
**Measurement:** amdgpu PPT (`/sys/class/hwmon/hwmon3/power1_average`) — µW, covers CPU+GPU+NPU package

| Condition       | Power (W) |
|-----------------|-----------|
| Idle baseline   | 23.682 |
| NPU loaded      | 37.615 |
| Delta (NPU+DMA) | 13.933 |

## Notes
- PPT = Package Power Tracking — whole SoC (CPU + GPU + NPU), no per-component isolation
- Delta includes host CPU overhead from the test binary's DMA dispatch loop
- Intel RAPL `energy_uj` exists at `/sys/class/powercap/intel-rapl:0/` but requires root
- Sample rate: 4 Hz (0.25s interval)
- Idle window: 6s (24 samples); loaded window: duration of 200-iter run
