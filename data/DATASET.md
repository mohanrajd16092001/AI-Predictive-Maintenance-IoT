# Synthetic IoT sensor dataset

This dataset is created by `src/data.py`. It contains no downloaded or measured industrial data. The document supplies the project topic and temperature, vibration, current and failure fields. All numeric generation formulas, timing, missingness, and label rules below are implementation assumptions.

Each of 80 fictional machines has one reading per simulated minute, beginning at an arbitrary fictional timestamp of 2026-01-01. Every history ends at a single generated failure event after 160 to 260 minutes. These timestamps do not describe a real experiment.

| Column | Meaning | Used as model input |
| --- | --- | --- |
| machine_id | Fictional machine identifier | No; split and replay only |
| timestamp | Fictional observation time | No; ordering and interpretation only |
| temperature | Celsius, invented degradation signal | Yes |
| vibration | Arbitrary simulated units; not a calibrated SW-420 reading | Yes |
| current | Amps, invented degradation signal | Yes |
| failure | 1 only at the terminal generated event | No |
| failure_next_10_readings | 1 when a failure appears in strictly the next 1 to 10 observations | Target only |

For each machine, `wear = elapsed_step / failure_step`. Temperature grows approximately with `wear**3`, vibration with `wear**4`, and current with `wear**2`, with invented baselines and Gaussian noise. About 1% of readings in each sensor column are set missing. Seed 42 makes generation repeatable in the recorded environment.

The generator knows the event time to construct labels. The predictor never receives that event time, wear, failure status, or future readings. Training excludes the terminal event itself. Missing sensor and interaction values are median-imputed using training data only.

This is deliberately a learnable educational generator. Its failure patterns are far simpler than a real machine's. All machines eventually fail; there are no censored histories, maintenance interventions, sensor calibration models, or independently collected validation data. Results demonstrate implementation behavior under these assumptions.

The terminal event has target 0 because there is no later event; it is excluded from modeling. Histories are fully observed through the event, so labels for preceding readings are known. The target concerns the next 10 simulated minutes and is binary, not an estimate of remaining useful life.
