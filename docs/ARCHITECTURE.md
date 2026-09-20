# Flybit architecture

Flybit is a desktop organism built around the public **Drosophila MaleCNS v1.0 connectome**.

## Non-negotiable control rule

Flybit must not use **semantic desktop shortcuts** as biological behaviour.
Application names, cursor identity, OCR labels or an LLM may never map directly
to movement.

Forbidden patterns include:

- `if mouse_near: escape()`
- `if food_exists: move_toward(food_xy)`
- LLM output mapped directly to movement
- application/window names selecting locomotion
- scripted behaviour presented as if it emerged from measured MaleCNS wiring

Flybit includes a clearly labelled **MODELED ethology/VNC bridge** because the
current simplified whole-CNS neuron model does not include complete VNC,
musculoskeletal, neuromodulatory and behavioural-selection physiology. v0.4.0
extends that bridge with explicit six-leg gait, take-off preload, flight
saccades, landing preparation, sleep stages, grooming targets, boundary
following and stable individual kinematic fingerprints. These layers may select
among modeled motor programmes only from non-semantic sensory quantities,
internal drives, local geometry and MaleCNS descending output.

The intended closed loop is now:

```
world
  -> sensory transduction
  -> MaleCNS / modeled sensory processing
  -> physiology + bounded sensory-context memory
  -> modeled ethology / VNC bridge
  -> motor -> body physics + six-leg proprioception
  -> world
```

Measured biology, modeled transduction/ethology and synthetic desktop-world
signals must remain explicitly separated in code and UI.

## v0.5 physical feedback and learning

- `simulation.py` owns the UI-independent ordered tick and returns a timestamped
  `SimulationSnapshot`; asynchronous neural outputs also carry monotonic step
  and timestamp data, with stale-latency telemetry.
- `proprioception.py` computes contact, touchdown, liftoff, load, slip, support
  identity and edge distance for all six legs. Landing completes only after a
  stable multi-leg contact; unstable contact produces abort/recovery events.
- `memory.py` stores at most 128 retinal/odor/geometry associations in a
  separately versioned atomic JSON file. Context similarity, decay and an
  eligibility trace produce small ethology biases, never motor commands.
- `compound_eye.py` maintains separate left/right fields across elevation bands
  and estimates horizontal/vertical flow, expansion, rotation and radial
  coherence while preserving the one-dimensional MaleCNS panorama.
- `world.py` exposes edge segments, corners, overlap depth, z-order and support
  continuity through `query_local_geometry`; titles are absent from this API.
- `sleep.py` represents long episodes with drowsy, light, deep and micro-awake
  stages and stage-specific noise, leakage and responsiveness gains.
- `assays.py` runs deterministic baseline, looming, food, edge, flight/landing,
  sleep/startle and learning scenarios and emits bounded JSON metrics.

### Evidence boundary

**MEASURED / DATA-DRIVEN:** MaleCNS neuron identities, connection direction and
weights, transmitter-derived signs and mapped photoreceptor identities.

**MODELED:** graded transfer dynamics, looming/motion transduction, ethology,
VNC decoding, learning, proprioception, gait, wing/body physics and sleep.
These are computational approximations and are not claimed as measured MaleCNS
biology.

**SYNTHETIC:** desktop surfaces and edges, virtual altitude, sugar/odor field,
care state and diagnostic/semantic telemetry. Semantic labels never enter the
behavioral control loop.

## Neural foundation

Flybit loads the same measured MaleCNS v1.0 wiring used by upstream `fly.ai`:

- 166,700 neurons
- 25,582,938 directed connections
- predicted neurotransmitter identity used for connection sign
- 6,006 mapped photoreceptors

## Mixed graded/spiking vision

The upstream all-LIF model loses raw eye signals at the first visual relay because real fly photoreceptors and early lamina monopolar cells do not communicate exclusively with action potentials.

Flybit therefore runs a mixed-signal network when `graded_visual=True`:

```
rendered luminance
    -> signed adapting contrast
    -> R1-6 / R7 / R8 continuous membrane state
    -> exact MaleCNS synapses
    -> L1 / L2 / L3 continuous membrane state
    -> exact MaleCNS synapses
    -> downstream spiking MaleCNS neurons
```

Only cell types with direct physiological support are placed in the graded set for now:

- R1-6
- R7
- R8
- L1
- L2
- L3

The analog output is a bounded signed function of membrane state. Hyperpolarization and depolarization therefore both alter downstream release instead of a hyperpolarized cell becoming permanently silent.

Connection identity, direction, weight and transmitter-derived sign still come from MaleCNS. There is no semantic object/threat/action classifier in this route. Flybit does include one explicit modeled optic-lobe boundary: frame-to-frame dark retinal expansion is converted into left/right looming drive on identified LPLC2 visual-projection cells. This compensates for motion selectivity that the simplified whole-CNS neuron model cannot be assumed to reproduce biophysically; it never stimulates descending motor neurons directly.

The graded time constants and transfer scale are computational parameters rather than fitted single-cell biophysical models. They must not be presented as experimentally measured membrane constants.

## Validation

Two validation levels protect the visual relay:

1. Synthetic unit tests verify the sign logic on a miniature photoreceptor -> L1 -> downstream circuit.
2. The `MaleCNS neural validation` workflow downloads the real prebuilt MaleCNS data and checks that raw scene contrast produces:
   - non-zero photoreceptor state,
   - non-zero L1/L2/L3 state,
   - non-zero graded synaptic current leaving the early visual relay.

A second real-MaleCNS validation checks the motor bridge itself: required descending groups must resolve, a single DNp01 spike must survive as an escape/take-off command, repeated DNa activity must produce locomotor drive, DNg02 subtype activity must produce flight thrust, and a raw-retina moving/expanding target must produce non-zero translational motor output.

## Desktop visual path

```
raw desktop pixels
    -> angular luminance panorama
    -> interpolation onto 6,006 photoreceptors
    -> graded R1-R8
    -> graded L1/L2/L3
    -> rest of MaleCNS
    -> descending-neuron motor read-out
    -> 2.5-D desktop body (screen x/y + independent altitude)
```

The screen sampler performs luminance downsampling only. It does not identify windows, text, objects or semantic threats. A separate temporal visual transducer compares consecutive raw panoramas to estimate retinal expansion and whole-field shift; the expansion channel is explicitly marked MODELED and targets LPLC2 rather than motor output. The cursor is rendered into the retinal panorama as a silhouette because platform screen capture commonly omits the hardware pointer.

The desktop retina targets a 20 ms cadence. To keep this practical, capture is restricted to the local region actually reached by the six retinal radii instead of copying the complete display. Because cursor angular width is computed from physical screen geometry, an approaching cursor occupies progressively more angular bins and therefore produces real temporal looming in the retinal stream.

A separate temporal observer computes cursor distance, velocity, acceleration, closing speed, angular growth, TTC and whole-field optic flow for validation/debugging. Cursor identity and semantic threat salience remain telemetry-only. The modeled near-field disturbance may modulate the explicitly modeled ethology layer, but it is not injected into MaleCNS as receptor-accurate mechanosensation.

v0.4.0 preserves a compact compound-eye field across multiple facet rows and
six sampling radii for temporal looming analysis. The brain-facing panorama
remains one-dimensional for MaleCNS compatibility, while the looming detector
requires angular, radial and cross-row coherence before producing a strong
expansion cue.

## Roadmap

1. **Neural foundation** — complete at the current MaleCNS abstraction level.
2. **Graded early vision** — implemented with real-data validation.
3. **Motor bridge** — identified descending groups plus side-preserving escape.
4. **Modeled ethology/VNC layer** — soft competing motor programs, sleep stages,
   grooming repertoire, foraging, feeding, boundary following and flight-state
   behaviour are active.
5. **Physical fly layer** — v0.4.0 adds explicit six-leg stance/swing gait,
   take-off preload, landing leg extension, flight saccades and gait/body
   coupling.
6. **Compound-eye fidelity** — multi-row local retinal field is active while the
   MaleCNS-facing panorama remains compatible with the existing photoreceptor
   mapping.
7. **Behaviour calibration** — runtime ethogram recording and deterministic CI
   calibration cover stop/walk/turn/boundary categories plus startle and landing.
8. **Next biological fidelity targets** — receptor-grounded olfaction,
   gustation, mechanosensation, proprioception and experimentally supportable
   mushroom-body plasticity.


## Desktop body bridge

The Windows organism is a compact 2.5-D physical model: screen x/y form the
desktop locomotion plane and flight uses an independent virtual altitude axis.

Measured/identified descending-neuron groups still provide the neural motor
signal:

- `DNg100` -> primary forward locomotor drive
- `DNa01` / `DNa02` -> steering and locomotor contribution
- `DNp01` -> escape / take-off impulse
- `DNg02_*` -> flight-thrust / wing-power drive
- `MDN` -> backward locomotor drive

v0.4.0 inserts explicit modeled body mechanics after those signals:

- six independently posed legs
- alternating modified-tripod stance/swing coordination
- speed-dependent stride frequency and stance fraction
- ground support coupled back into planar motion
- a short leg preload before escape take-off
- virtual-altitude lift/gravity flight
- brief yaw saccades while airborne
- landing drive that reduces lift, slows translation and extends all six legs
- substrate identity and geometric boundary cues from visible windows/taskbar

The renderer consumes the same leg poses used by the body model, so walking,
landing and grooming are no longer unrelated decorative animations.

The mapping from descending-neuron firing to body forces remains a model-defined
decoder, not a complete musculoskeletal Drosophila simulation. That distinction
is surfaced in the UI and documentation.


## Desktop UI

Normal mode contains only a small always-on-top fly overlay. Clicking the organism opens a native resizable Windows neural control panel. The panel displays:

- sampled MaleCNS EM-position brain map
- current spiking activity overlay
- R1-R8 / L1-L3 graded visual telemetry
- DNg100/DNa walking, DNa steering, DNp01 escape, DNg02 flight and MDN backward channels
- Care state and physical sugar placement
- short neural/body event log

The panel does not provide direct movement commands. Feeding changes persistent nutritional state only after physical contact with the desktop sugar object. The food object is visible to the retinal sampler, but hunger is not converted into scripted navigation.

## Care and feeding

Flybit maintains app-level nutritional state: hunger, feeding count and last-feed timestamp. A sugar drop is a real desktop-world object with position and collision radius. Consumption occurs only when the body overlaps the drop.

This is deliberately separated from neural claims. The bundled MaleCNS metadata used by Flybit does not currently expose receptor-level sweet-GRN identity, so Flybit does not pretend to inject a biologically exact sugar taste signal. When a validated receptor/cell mapping is available in the bundled data, gustatory transduction can be added without changing this physical contact model.


## Semantic perception boundary

Flybit now maintains a separate OS-level semantic perception stream for coarse
desktop entities: hardware cursor, top-level windows, native Win32 buttons and
recognized application processes such as Chrome. These labels are displayed in
telemetry and may provide future sensory context, but they are not action
commands and are not allowed to bypass the MaleCNS/body loop.

## Homeostasis and life history

Hunger is represented as an internal homeostatic drive. It modulates locomotor
readiness symmetrically rather than selecting a target or turn direction.
Feeding requires physical contact, reduces hunger, emits a short feeding signal
and restores metabolic energy.

Each persistent organism has a wall-clock birth timestamp, deterministic
individual phenotype, finite modeled lifespan, metabolic energy and vitality.
Activity, boldness and curiosity now modulate global neural tonic/noise/arousal
statistics symmetrically; they do not directly choose steering, escape or
feeding actions.
Ageing and energy can reduce body capacity. These are computational organism
states; they must not be described as literal biological immortality or a
complete living animal.

## Biomechanical telemetry

The 2-D body now exposes speed, acceleration, turn rate, gait phase, estimated
wingbeat rate and locomotor load. This does not yet replace the kinematic body
with a full musculoskeletal Drosophila model, but it creates the instrumentation
needed to validate that future biomechanical layer.


## Persistent organism identity

Each organism may have a compact persistent display name. The name is cosmetic
identity only: it has no neural or behavioural effect. It can be edited only
from the native control panel's Life tab; the fly context menu does not expose
a rename action.


## Circadian rest physiology

Flybit maintains a modeled wake drive with broad morning/evening activity peaks,
ambient-retina luminance input and persistent homeostatic sleep pressure. The
resulting rest drive modulates global neural tonic activity, intrinsic noise and
locomotor readiness. It never calls a sleep, wake, turn or escape action
directly. The clock shape and timescales are computational approximations rather
than a complete emulation of the Drosophila circadian network.


## Modeled food odor field

The desktop sugar source now emits a smooth synthetic odor field sampled at two
virtual antenna positions. Hunger changes the observer salience of that odor but
does not change its physical concentration. Left/right concentration and
gradient are exposed in the Care panel. The odor signal is intentionally not
injected into MaleCNS yet: the bundled runtime metadata does not provide the
receptor-level olfactory mapping needed to make that claim responsibly.


## Habituation and sensitization

Repeated raw retinal expansion gradually reduces the modeled looming-transducer
gain and recovers when expansion stops. Strong stimuli retain a non-zero response.
This is short-term sensory habituation, not a motor policy.

Conversely, when the nervous system itself produces DN escape output, a
short-lived threat-arousal state rises and decays over seconds. That state
modulates global neural tonic/noise/readiness only; it cannot trigger an escape
on its own. This gives post-escape sensitization without a mouse-near rule.


## Native window substrate contact

Visible external top-level Win32 windows are periodically scanned as rectangular
substrates while preserving their native Z-order. When Flybit is grounded, the
topmost rectangle under its current desktop coordinate is recorded as physical
contact context; otherwise the support is the desktop plane. While airborne the
support is explicitly Air.

This contact model does not apply app-specific friction, attraction, avoidance
or steering. Window titles are observer metadata only and cannot choose
behaviour. The geometry exists so landing/contact telemetry follows the current
desktop instead of the old unused surface placeholder.
