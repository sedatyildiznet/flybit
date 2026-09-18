# Flybit

<p align="center">
  <strong>A connectome-driven desktop organism for Windows.</strong><br>
  Flybit turns the Drosophila MaleCNS connectome into a persistent creature that sees, reacts, moves, rests and survives on your desktop.
</p>

<p align="center">
  <a href="https://github.com/sedatyildiznet/flybit/releases/tag/v0.3.0"><img alt="Release" src="https://img.shields.io/badge/release-v0.3.0-2ea44f"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-stable-2ea44f">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-0078D4">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

## What is Flybit?

Flybit is an experimental **desktop organism** built around the public **Drosophila MaleCNS v1.0 connectome**.

It is not a scripted desktop pet and it is not an LLM choosing actions for a sprite.

The central rule of the project is:

> **Application code must not choose biological behaviour on behalf of the organism.**

The intended control loop is always:

```text
DESKTOP WORLD
      ↓
SENSORY TRANSDUCTION
      ↓
MALECNS NEURAL DYNAMICS
      ↓
DESCENDING / MOTOR OUTPUT
      ↓
BODY PHYSICS
      ↓
DESKTOP WORLD
```

Flybit therefore avoids shortcuts such as:

```python
if mouse_is_near:
    escape()

if hungry:
    move_to_food()
```

Mouse approach, light, movement, hunger, fatigue and other state changes must influence the organism through sensory or physiological layers rather than directly selecting the action.

---

## v0.3.0 Stable

**v0.3.0 is the first stable Flybit release.**

This release combines the neural desktop foundation with the persistent-organism layer introduced during the alpha series.

Major additions include:

- 50 Hz target desktop retinal sampling
- raw-luminance temporal looming
- LPLC2 visual-motion transduction
- cursor velocity, acceleration, closing-speed and TTC telemetry
- optic-flow observation
- short-term visual habituation
- escape-driven sensitization
- persistent neural membrane/adaptation state
- 2.5-D flight with an independent altitude axis
- grounded tripod-gait rendering
- native Win32 window substrate/contact awareness
- hunger, energy, ageing and individual phenotype
- modeled circadian wake/rest physiology
- bilateral synthetic food-odor telemetry
- persistent organism naming from the control panel
- explicit **Measured / Modeled / Synthetic** provenance in the UI
- real-MaleCNS validation in CI

Latest stable release:

**[Download Flybit v0.3.0](https://github.com/sedatyildiznet/flybit/releases/tag/v0.3.0)**

---

## Why this project exists

A normal desktop pet follows authored behaviour:

```text
timer -> random choice -> animation
```

Flybit is trying to explore a different idea:

```text
environment -> senses -> nervous system -> motor output -> body
```

The goal is not to fake intelligence with more state-machine branches. The goal is to make the organism increasingly dependent on the same kind of closed sensory-motor loop that makes a biological animal responsive to its environment.

That does **not** mean Flybit is a literal living fly or a complete brain emulation. It means the project deliberately preserves the distinction between:

- measured biological data,
- computational models,
- synthetic desktop-world physics,
- and behaviour that actually emerges from the neural loop.

---

# Core system

## MaleCNS neural foundation

Flybit loads the MaleCNS network used by the upstream `fly.ai` project.

Current runtime scale:

| Component | Current Flybit model |
|---|---:|
| Neurons | **166,700** |
| Directed connectome connections | **25,582,938** |
| Mapped photoreceptors | **6,006** |
| Desktop retinal panorama | **384 angular bins** |
| Retinal sampling target | **20 ms / 50 Hz** |

Connection identity, direction, weight and transmitter-derived sign remain grounded in MaleCNS data.

The desktop-organism path does **not** rely on the upstream task-oriented `FeatureDetectors` shortcut for choosing movement.

---

## Mixed graded + spiking vision

Real fly photoreceptors and several early visual neurons are not well represented as ordinary all-or-nothing spiking cells.

Flybit therefore uses a mixed visual model:

```text
desktop luminance
      ↓
signed adapting contrast
      ↓
R1-6 / R7 / R8 graded state
      ↓
measured MaleCNS connectivity
      ↓
L1 / L2 / L3 graded state
      ↓
measured MaleCNS connectivity
      ↓
downstream spiking network
```

Currently modeled as graded cells:

- R1-6
- R7
- R8
- L1
- L2
- L3

The graded membrane constants and transfer functions are computational parameters. They are **not presented as fitted single-cell biophysical models**.

---

## Desktop retina

Flybit continuously samples the desktop around its own body.

Instead of repeatedly copying the entire monitor, v0.3.0 captures only the local retinal neighbourhood required by the six visual sampling radii.

Visible desktop content can therefore affect retinal input directly:

- windows
- icons
- text
- images
- video
- motion
- the hardware cursor
- the sugar object

The operating-system cursor is rendered into the retinal panorama because normal desktop capture frequently omits the hardware pointer.

---

## Looming and threat perception

Flybit does not internally receive the command:

> "The mouse is near. Escape."

Instead, temporal visual information is calculated from consecutive retinal observations.

The sensory observer tracks values such as:

- cursor distance
- cursor speed
- cursor acceleration
- relative closing speed
- angular size
- angular expansion
- time-to-collision
- whole-field optic flow

More importantly, raw luminance frames are compared for **retinal expansion**.

```text
raw retinal frames
      ↓
dark expansion / looming estimate
      ↓
left / right modeled visual-motion cue
      ↓
identified LPLC2 visual-projection cells
      ↓
MaleCNS network
      ↓
descending activity
      ↓
motor decoder
```

The modeled boundary targets **LPLC2 visual projection cells**, not descending motor neurons.

Current MaleCNS validation resolves:

- **94 left LPLC2 cells**
- **91 right LPLC2 cells**

Live LPLC2 activity is visible in the control panel.

---

## Habituation and sensitization

Flybit contains simple short-term adaptive state without turning it into a scripted behaviour tree.

### Habituation

Repeated harmless retinal expansion gradually reduces the modeled looming-transducer gain.

When the stimulus disappears, the response recovers.

A strong stimulus still retains a minimum response.

### Sensitization

When the nervous system itself produces DN escape output, Flybit raises a temporary threat-arousal state.

That state changes general neural readiness; it does not trigger another escape by itself.

```text
real DN escape output
      ↓
temporary threat arousal
      ↓
global tonic / noise / readiness modulation
```

---

# Lifelike ethology layer

v0.3.0 adds an explicit **MODELED ethology/VNC bridge** between descending
neural output and the desktop body. This exists because a simplified whole-CNS
LIF model plus a small descending-neuron decoder does not reproduce the full
behavioural repertoire of a living fly by itself.

The layer never receives application names or semantic commands such as
`mouse_near`, `seek_food` or `go_left`. It operates on sensory and internal
quantities already present in the organism:

- left/right retinal expansion
- modeled near-field disturbance
- bilateral food odor concentration and gradient
- hunger
- sleep/rest pressure
- threat arousal
- persistent activity/boldness/curiosity traits
- MaleCNS descending motor output

Competing semi-Markov motor programmes now include:

- idle / stop
- walking bouts
- short turn bouts
- odor-guided foraging
- physical feeding bouts
- grooming
- quiescent sleep
- directional escape
- stabilized post-takeoff flight

MaleCNS motor output is preserved and can override quieter modeled programmes.
Strong looming can also interrupt sleep, grooming or feeding immediately.

This layer is intentionally labelled **MODELED**, not measured MaleCNS
physiology. Its purpose is to supply missing behavioural/VNC/body coupling
without pretending the current connectome simulator already contains every
biophysical mechanism required for natural behaviour.

---

# Motor system

## Identified descending outputs

The desktop body is driven from identified MaleCNS motor-related groups.

| Neural group | Flybit interpretation |
|---|---|
| `DNg100` | primary forward locomotor drive |
| `DNa01 / DNa02` | steering + bilateral locomotor contribution |
| `DNp01` | escape / take-off impulse |
| `DNg02_*` | sustained flight / wing-power drive |
| `MDN` | backward locomotion |

Most groups are converted into short-window firing-rate estimates before force mapping.

This prevents a single stochastic spike from turning into a large body movement.

`DNp01` is intentionally different: a single detected Giant Fiber / DNp01 event is preserved as an immediate escape/take-off signal.

---

## 2.5-D body

Flybit is no longer modeled as a sprite that simply moves faster when "airborne".

The body has:

```text
x
y
vx
vy
heading
angular velocity
altitude
vertical velocity
airborne state
```

Screen `x/y` remain desktop coordinates.

Flight uses a **separate virtual altitude axis**.

This matters because monitor Y is not treated as biological height. Gravity therefore never makes Flybit fall toward the bottom edge of the screen.

### Flight sequence

```text
DNp01 escape activity
      ↓
take-off impulse
      ↓
virtual altitude rises
      ↓
DNg02 contributes sustained lift
      ↓
flight
      ↓
lift decays
      ↓
altitude returns to desktop plane
      ↓
landing
```

---

## Walking and gait

Grounded movement exposes a locomotor gait phase.

The renderer uses that phase to animate an alternating tripod-style leg pattern.

This gait rendering does not choose movement. It reflects body state that already came from the neural/motor loop.

Biomechanical telemetry includes:

- speed
- acceleration
- turn rate
- gait phase
- wingbeat estimate
- locomotor load
- altitude
- vertical speed
- current substrate

---

# The desktop as a world

## Native window substrates

Flybit scans visible top-level Win32 windows and preserves their native Z-order.

When grounded, the organism knows which physical desktop substrate is under its body:

```text
Air
Window
Desktop
```

Overlapping windows use the topmost visible rectangle.

This information is **physical contact context only**.

Flybit does not become attracted to Chrome because a window title says "Chrome", and window names never issue steering commands.

---

## Semantic perception

Flybit also maintains a separate observer/debug perception stream for coarse OS-level objects such as:

- cursor
- top-level windows
- native buttons
- known application processes

These labels are useful for the human-readable control panel and diagnostics.

They do not bypass the nervous system.

The fly may therefore be displayed as being near a Chrome window while its biological visual path still receives only physical visual quantities such as luminance and motion.

---

# Organism state

## Persistent identity

Flybit is intended to remain the same individual between launches.

Persistent state currently includes parts of:

- birth timestamp
- organism name
- desktop position
- heading
- panel geometry
- hunger
- feeding history
- energy
- sleep pressure
- neural membrane state
- retinal adaptation
- recent motor-rate state

The organism name is cosmetic only.

It can be changed **only from the Life tab of the control panel**.

---

## Neural state persistence

Flybit periodically saves dynamic neural state to disk.

Persisted neural information includes:

- membrane values
- retinal adaptation
- short motor-rate history
- neural simulation step state

This does **not** mean Flybit currently has full long-term synaptic learning.

Connectome weights are not silently rewritten and the project does not claim mushroom-body plasticity where none has been implemented.

---

## Hunger and energy

Hunger is an internal homeostatic variable.

It can affect general neural/locomotor readiness, but it does not contain a hidden food-navigation command.

Flybit also tracks:

- metabolic energy
- vitality
- feeding count
- last feeding time
- age
- finite modeled lifespan

Age and energy can reduce movement capacity.

---

## Individual phenotype

Each persistent organism has stable modeled traits:

- activity
- boldness
- curiosity

These values modulate global neural statistics such as tonic activity, intrinsic noise and arousal.

They do not directly mean:

```text
bold -> attack
curious -> move right
active -> wander
```

Two Flybit individuals can therefore differ in neural readiness without personality variables becoming a hidden behaviour script.

---

# Feeding

## Physical sugar placement

The **Care** tab lets the user place a sugar drop anywhere on the desktop.

The organism must physically contact it before the feeding event occurs.

```text
place sugar
      ↓
visible world object
      ↓
retinal visibility
      ↓
physical body contact
      ↓
consumption
      ↓
hunger / energy update
```

Feeding count and last-feed time persist across launches.

---

## Modeled food odor

v0.3.0 also includes a synthetic bilateral odor field around the sugar source.

Two virtual antenna positions sample:

- left concentration
- right concentration
- bilateral gradient
- mean concentration
- hunger-dependent salience

This is deliberately labeled **MODELED / SYNTHETIC**.

The signal is **not injected into MaleCNS olfactory neurons**, because the bundled runtime metadata does not currently provide the receptor-level mapping required to make that claim responsibly.

In v0.3.0 the bilateral odor field instead feeds the explicitly modeled
ethology/VNC bridge. A hungry organism can therefore perform gradient-based
foraging without pretending that the synthetic odor field is receptor-accurate
MaleCNS physiology. Physical body contact is still required before feeding.

The same scientific boundary applies to receptor-accurate gustation and
mechanosensation.

---

# Circadian rest

Flybit maintains a modeled wake/rest physiology using local time, ambient
retinal luminance, homeostatic sleep pressure and recent locomotor load.

v0.3.0 converts that physiology into explicit **quiescent sleep bouts** rather
than only reducing tonic/noise values. Weak stochastic motor leakage is gated
during sleep, while sufficiently strong retinal looming can wake the organism
straight into an escape response.

Offline time now advances hunger, low-activity metabolism and sleep homeostasis,
so closing the application no longer freezes every physiological variable while
wall-clock age continues to advance.

---

# Control panel

Click Flybit to open the native resizable control panel.

The panel is diagnostic and care-oriented; it does not contain manual movement controls.

## Overview

General neural/body activity and current state.

## Brain

Live neural map and spiking activity.

## Care

- hunger
- feeding history
- sugar placement
- modeled bilateral odor field

## Events

Short neural/body event history including take-off, landing and substrate changes.

## Perception

Live diagnostics such as:

- cursor distance
- cursor velocity
- closing speed
- TTC
- optic flow
- left/right retinal looming
- habituation
- modeled near-field disturbance
- observer threat salience

## Life

- persistent organism name
- age
- lifespan
- energy
- vitality
- activity
- boldness
- curiosity
- sleep pressure
- circadian wake/rest state
- biomechanical telemetry

## Model

The Model tab makes the project's scientific boundary explicit.

### MEASURED / DATA-DRIVEN

Examples:

- MaleCNS neuron identities
- connectome graph
- connection directions
- synapse-derived weights
- transmitter-derived signs
- mapped photoreceptors
- identified DN / LPLC2 cell types

### MODELED

Examples:

- graded transfer dynamics
- retinal looming -> LPLC2 transduction
- descending-neuron -> body force decoding
- altitude / lift / gravity
- gait rendering
- hunger / circadian / phenotype modulation

### SYNTHETIC WORLD / TELEMETRY

Examples:

- desktop sugar object
- odor field
- semantic window/application labels
- TTC observer
- threat-salience observer
- near-field disturbance observer

### NOT CLAIMED

Flybit does not currently claim:

- complete biological brain emulation
- full musculoskeletal Drosophila physics
- receptor-accurate taste
- receptor-accurate olfaction
- receptor-accurate mechanosensation
- complete hormonal physiology
- complete synaptic plasticity
- literal artificial life in the biological sense

---

# Installation

## Windows stable release

Use the latest stable release:

**[Flybit v0.3.0](https://github.com/sedatyildiznet/flybit/releases/tag/v0.3.0)**

The Windows package is produced by GitHub Actions as:

```text
Flybit-Windows-x64.zip
└── Flybit.exe
```

The MaleCNS-derived runtime data is distributed separately from the executable. On first launch the neural core may need to download roughly **260 MB** of prebuilt data.

---

## Run from source

### Requirements

- Python 3.10+
- Windows for the full desktop-organism experience
- Git
- internet access on the first neural-data setup

### Setup

```powershell
git clone https://github.com/sedatyildiznet/flybit.git
cd flybit

python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip
pip install -e ".[desktop]"

python -m flybit
```

---

# Validation

Flybit includes both deterministic unit tests and validation against real MaleCNS-derived runtime data.

## Unit tests

```bash
python -m unittest discover -s tests -v
```

Coverage includes areas such as:

- hunger and physical feeding
- circadian pressure
- neural visual relay
- life/energy state
- motor/body dynamics
- altitude take-off and landing
- substrate contact
- olfactory-world model
- cursor dynamics
- optic flow
- retinal looming
- habituation
- persistent identity

---

## Real MaleCNS visual validation

```bash
python scripts/validate_graded_vision.py
```

The validation requires the real early visual relay to produce:

- non-zero photoreceptor state
- non-zero L1/L2/L3 state
- non-zero graded current into downstream neurons
- visual-projection spiking activity

---

## Real MaleCNS motor / looming validation

```bash
python scripts/validate_motor_groups.py
```

The current validation checks that:

- required descending motor groups resolve
- left/right LPLC2 groups resolve
- one DNp01 event survives the motor decoder as escape/take-off
- bilateral DNa activity produces locomotor drive
- DNg02 activity produces flight drive
- raw retinal stimulation reaches translational motor output
- expanding retinal input is distinguishable from a static silhouette
- expanding input increases LPLC2 activity
- the expanding challenge reaches DNp01 escape output

These validations are also exercised through GitHub Actions.

---

# Project architecture

At a high level:

```text
┌───────────────────────────────────────────────┐
│               DESKTOP WORLD                   │
│ windows · pixels · cursor · food · time       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 SENSORY LAYER                 │
│ retina · temporal loom · optic flow           │
│ modeled odor · observer telemetry             │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                  MALECNS                      │
│ graded early vision + spiking CNS             │
│ measured connectome connectivity              │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│             DESCENDING OUTPUT                 │
│ DNg100 · DNa01/02 · DNp01 · DNg02 · MDN      │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│       MODELED ETHOLOGY / VNC BRIDGE           │
│ bouts · sleep · groom · forage · escape       │
└───────────────────────┬───────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────┐
│                 BODY MODEL                    │
│ x/y · heading · gait · altitude · flight      │
└───────────────────────┬───────────────────────┘
                        │
                        └──────────────► WORLD
```

For implementation details and scientific boundaries, see:

**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

---

# What Flybit deliberately does not do

Flybit intentionally avoids several tempting shortcuts.

It does not:

- use an LLM as the fly's brain
- use computer vision labels to directly select movement
- run a random-walk routine and call it neural behaviour
- call `escape()` merely because the cursor is close
- call `seek_food()` merely because hunger is high
- pretend synthetic odor is measured olfactory physiology
- present modeled membrane constants as measured physiology
- claim a complete biological fly simulation

This constraint is one of the main design goals of the project.

---

# Roadmap

v0.3.0 establishes the first lifelike behavioural layer on top of the persistent
MaleCNS organism.

### Implemented in v0.3.0

- stochastic stop / walk / turn bouts
- side-preserving escape steering
- pre-escape threat sensitization
- explicit sleep and wake-by-startle
- grooming bouts and matching body animation
- bilateral odor-guided foraging
- stationary feeding bouts
- offline hunger / metabolism / sleep progression
- one world/physics clock using measured frame `dt`
- multi-radius retinal samples for more coherent looming detection

### Next fidelity targets

- receptor-grounded olfactory and gustatory injection where validated mappings exist
- richer six-leg/VNC biomechanics and proprioceptive feedback
- contact mechanosensation
- more realistic flight stabilization and landing
- experimentally supportable mushroom-body plasticity
- persistent learned food/threat associations
- behavioural assay distributions matched against published Drosophila data
- a fuller compound-eye spatial model beyond the current brain-facing 1-D panorama

The rule remains the same: measured biology, modeled transduction and synthetic
desktop-world mechanisms must remain clearly separated.

---

# Scientific provenance

Flybit is derived from **[alextitonis/fly.ai](https://github.com/alextitonis/fly.ai)**.

The MaleCNS v1.0 connectome was produced by FlyEM / HHMI Janelia and collaborators and is distributed separately under its own terms.

Flybit does not redistribute the raw MaleCNS connectome in this repository.

See **[NOTICE.md](NOTICE.md)** for attribution details.

---

# License

Flybit code is distributed under the **MIT License** unless a file states otherwise.

MaleCNS data remains under its original data license.

See **[LICENSE](LICENSE)**.
