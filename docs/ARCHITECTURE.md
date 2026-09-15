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

The test deliberately does not require a specific behavioural output. Behaviour should only be claimed after the motor/body loop is connected and validated.

## Desktop visual path

```
raw desktop pixels
    -> angular luminance panorama
    -> interpolation onto 6,006 photoreceptors
    -> graded R1-R8
    -> graded L1/L2/L3
    -> rest of MaleCNS
    -> descending-neuron motor read-out
    -> flat 2-D desktop body
```

The screen sampler performs luminance downsampling only. It does not identify windows, text, objects, motion classes or threats. The cursor is rendered into the retinal panorama as a silhouette because platform screen capture commonly omits the hardware pointer.

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

The current Windows organism uses a deliberately small 2-D kinematic decoder. It does not inspect the cursor or window state to select behaviour.

Measured/identified descending-neuron groups provide the motor signal:

- `DNg100` -> forward locomotor drive
- `DNa02` -> differential steering
- `DNp01` -> escape / take-off impulse
- `MDN` -> backward locomotor drive

The whole desktop is one flat locomotion plane. The body decoder has planar drag and desktop boundaries, but no downward gravity, falling or window-edge landing physics.

DNp01 starts a short planar flight burst rather than a fake vertical jump. DNa02 is decoded as yaw-only steering, so the rendered body cannot roll or pitch into somersaults.

Descending motor groups are converted to short-window firing-rate estimates before force mapping. This prevents isolated stochastic spikes from becoming full movement commands while preserving sustained neural activity.

Cursor approach and all other visible desktop content enter through the retinal luminance panorama. There is no `mouse_near -> escape` rule.

The mapping from descending-neuron firing to 2-D forces is still a model-defined motor decoder, not a complete fly musculoskeletal model. That distinction is surfaced in the UI and documentation.

## Desktop UI

Normal mode contains only a small always-on-top fly overlay. Clicking the organism opens a native resizable Windows neural control panel. The panel displays:

- sampled MaleCNS EM-position brain map
- current spiking activity overlay
- R1-R8 / L1-L3 graded visual telemetry
- DNg100, DNa02, DNp01 and MDN motor channels
- short neural/body event log

The panel is observational. It does not provide movement commands.
