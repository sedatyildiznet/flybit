# Flybit architecture

Flybit is a desktop organism built around the public **Drosophila MaleCNS v1.0 connectome**.

## Non-negotiable control rule

Flybit must not contain application logic that chooses biological behaviour on behalf of the nervous system.

Forbidden patterns include:

- `if mouse_near: escape()`
- LLM output mapped directly to movement
- scripted wandering presented as neural behaviour
- personality sliders that directly select actions

The intended closed loop is:

```
world -> sensory transduction -> MaleCNS neural dynamics -> motor nervous system -> body physics -> world
```

Code outside the neural/body model may translate physical quantities between domains, but it must not select the action.

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

Connection identity, direction, weight and transmitter-derived sign still come from MaleCNS. There is no object, looming, threat or action classifier in this route.

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

The screen sampler performs luminance downsampling only. It does not identify windows, text, objects, motion classes or threats. The cursor is rendered into the retinal panorama as a silhouette because platform screen capture commonly omits the hardware pointer.

The desktop retina targets a 20 ms cadence. To keep this practical, capture is restricted to the local region actually reached by the six retinal radii instead of copying the complete display. Because cursor angular width is computed from physical screen geometry, an approaching cursor occupies progressively more angular bins and therefore produces real temporal looming in the retinal stream.

A separate temporal observer computes cursor distance, velocity, acceleration, closing speed, angular growth, TTC and whole-field optic flow for validation/debugging. These values are not movement commands. A modeled near-field disturbance value is also exposed but remains telemetry-only until a defensible MaleCNS mechanosensory receptor mapping is available.

## Roadmap

1. **Neural foundation** — complete.
2. **Graded early vision** — implemented; real-MaleCNS validation required before release.
3. **Motor bridge** — identified descending-neuron groups drive the current 2-D desktop body.
4. **Closed-loop desktop world** — raw screen luminance and planar body physics feed each other continuously; fuller biomechanical coupling remains future work.
5. **Feeding / mechanosensation** — olfactory, gustatory and mechanosensory transduction from published biology.
6. **Plasticity** — mushroom-body dopamine-gated learning where experimentally supportable.
7. **Persistent individual** — persist neural/plastic state across launches.

## Scientific honesty

MaleCNS is a measured wiring diagram. Flybit is still a computational nervous-system model, not a complete biological brain emulation. Receptor-specific effects, many graded cell types, neuromodulation, detailed dendritic integration, muscle/body coupling and plasticity remain incomplete.


## Desktop body bridge

The current Windows organism uses a deliberately small 2.5-D kinematic decoder. It does not inspect the cursor or window state to select behaviour.

Measured/identified descending-neuron groups provide the motor signal:

- `DNg100` -> primary forward locomotor drive
- `DNa01` / `DNa02` -> differential steering plus bilateral locomotor contribution
- `DNp01` -> escape / take-off impulse; a single detected spike is preserved
- `DNg02_*` MaleCNS subtypes -> flight-thrust / wing-power drive
- `MDN` -> backward locomotor drive

Screen x/y form one flat locomotion plane. Flight now has a separate virtual altitude and vertical-velocity axis. DNp01 supplies a take-off impulse and DNg02 contributes sustained lift; gravity acts only on this virtual altitude, never on monitor Y. Landing occurs when altitude returns to the desktop plane. This preserves the no-falling-across-the-monitor rule while giving airborne state physical duration.

DNa02 is decoded as yaw-only steering, so the rendered body cannot roll or pitch into somersaults. Grounded rendering derives an alternating tripod gait from locomotor phase; the gait is presentation/body coupling and does not choose direction.

Most descending motor groups are converted to short-window firing-rate estimates before force mapping. This prevents isolated stochastic spikes from becoming full movement commands while preserving sustained neural activity. DNp01 is the exception: Giant Fiber take-off physiology is event-like, so one DNp01 spike is preserved as an immediate escape signal.

Cursor approach and all other visible desktop content enter through the retinal luminance panorama. There is no `mouse_near -> escape` rule.

The mapping from descending-neuron firing to 2-D forces is still a model-defined motor decoder, not a complete fly musculoskeletal model. That distinction is surfaced in the UI and documentation.

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
