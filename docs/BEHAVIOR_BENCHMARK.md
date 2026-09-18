# Flybit behaviour benchmark

Flybit v0.4.0 records a runtime ethogram without influencing behaviour.

The recorder stores:

- time spent in each behaviour mode
- transition count
- mean bout duration by mode
- mean speed
- mean absolute turn rate
- speed coefficient of variation

For locomotion analysis, Flybit modes can be projected onto the common
**Stop / Curved Walk / Sharp Turn / Boundary (WTSB)** categories used in
walking-Drosophila studies.

The CI script `scripts/validate_ethogram.py` runs a deterministic virtual
organism and checks that:

1. neutral life produces multiple locomotor/behavioural modes rather than one
   dominant scripted loop;
2. walk, turn, boundary and stop-like states are represented;
3. hungry odor exposure produces foraging;
4. rest pressure produces sleep;
5. strong grounded looming produces immediate escape;
6. coherent airborne looming produces landing preparation;
7. transition diversity and speed variability remain above minimum calibration
   thresholds.

This is an **engineering calibration suite**, not a percentage claim that Flybit
is biologically identical to a real fly. Exact distribution matching requires a
curated experimental trajectory dataset with equivalent environmental
conditions.

Scientific design references include:

- Yang et al., *Sensorimotor transformation underlying odor-modulated locomotion
  in walking Drosophila*, Nature Communications (2023).
- Mendes et al., *Drosophila uses a tripod gait across all walking speeds*,
  eLife / PMC (2021).
- Seeds et al., *A suppression hierarchy among competing motor programs drives
  sequential grooming in Drosophila*, eLife (2014).

The repository keeps measured connectome data, modeled ethology/body dynamics
and synthetic desktop-world geometry explicitly separated.
