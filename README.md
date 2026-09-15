# Flybit

**Flybit is a desktop organism powered by the Drosophila MaleCNS connectome.**

The project is built from [alextitonis/fly.ai](https://github.com/alextitonis/fly.ai) and keeps one rule above everything else:

> The application must not choose biological behaviour for the fly.

The intended loop is:

```
desktop world
  -> sensory transduction
  -> MaleCNS neural dynamics
  -> motor nervous system
  -> body physics
  -> desktop world
```

No `if mouse_near: escape()`, no scripted wandering presented as neural behaviour, and no LLM directly controlling movement.

## Current state

Flybit currently:

- loads the full MaleCNS network used by `fly.ai`
- runs 166,700 neurons and 25,582,938 measured connectome connections
- projects the desktop scene into 6,006 MaleCNS photoreceptors
- models R1-6, R7, R8 and L1/L2/L3 as graded visual cells
- propagates their signed analog state through the measured MaleCNS connections
- keeps the remaining network on the upstream spiking dynamics
- does **not** use the `FeatureDetectors` shortcut in the desktop organism path
- exposes retina, lamina, visual-projection and descending-neuron telemetry
- decodes identified descending neurons into a 2-D desktop body: DNg100 + DNa01/DNa02 walking, DNa01/DNa02 steering, DNp01 escape/take-off, DNg02 flight thrust and MDN backward
- treats screen x/y as a flat desktop locomotion plane and uses a separate virtual altitude axis for take-off/flight/landing; gravity never pulls the fly toward the bottom of the monitor
- shows only the fly during normal use; clicking it opens a live neural control panel with brain map, motor channels and short logs
- migrates old edge-stranded state safely, then persists desktop position/heading and control-panel geometry across launches
- samples raw desktop pixels rather than only tracking the cursor
- runs the desktop retina on a 20 ms target cadence using a local retinal capture region instead of copying the full screen
- tracks temporal cursor kinematics (distance, speed, acceleration, closing speed, angular growth and time-to-collision) plus whole-field optic flow as sensory/debug telemetry
- lets the cursor create genuine retinal looming through its changing angular extent; no mouse-distance escape rule is used
- smooths descending-neuron firing into short-window motor rates to prevent single-spike spin artifacts
- includes a branded application/EXE icon and a native resizable tabbed Windows control panel
- includes persistent Care state and physical sugar-drop feeding
- exposes hunger as an internal homeostatic drive that modulates neural locomotor readiness without choosing direction
- tracks wall-clock age, finite modeled lifespan, metabolic energy, vitality and stable individual phenotype
- makes activity, boldness and curiosity alter global neural arousal/intrinsic activity without directly choosing a direction or action
- persists a user-chosen organism name; naming is available only from the Life tab of the control panel
- maintains modeled circadian wake drive and persistent sleep pressure; these reduce/increase global neural arousal instead of issuing a scripted sleep command
- adds semantic desktop perception for cursor, windows, native buttons and recognized applications such as Chrome
- reports biomechanical telemetry including speed, acceleration, turn rate, gait phase, wingbeat rate and locomotor load
- builds a Windows executable through GitHub Actions
- runs a separate real-MaleCNS graded-vision validation workflow

The mixed visual model fixes a structural limitation of the upstream all-spiking approximation, but it is still not a complete biophysical simulation. The graded time constants and transfer scale are computational parameters, not fitted membrane models.

## Windows

Every published GitHub Release triggers the Windows workflow and attaches:

```
Flybit-Windows-x64.zip
└── Flybit.exe
```

The connectome data is not bundled into the executable. On first launch, the neural core may download about 260 MB of prebuilt MaleCNS-derived data.

## Run from source

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[desktop]"
python -m flybit
```

During normal use Flybit appears as a small always-on-top fly moving across the desktop plane. The screen is sampled as raw luminance rays around the fly (384 angular bins across six radii) on a 20 ms target cadence and interpolated onto the MaleCNS photoreceptors. The sampler captures only the local retinal neighbourhood needed by those rays, avoiding repeated full-screen copies. The retinal path itself still has no OCR or threat classifier. A separate OS-level semantic perception layer labels coarse desktop entities (cursor, windows, native Win32 buttons and common applications) for sensory context and telemetry; labels never map directly to movement commands.

Windows, icons, text, images, video and other visible screen content can therefore change retinal input directly. The hardware cursor is added as a retinal silhouette because normal screen capture often omits it. Its angular extent grows naturally as it approaches Flybit, producing a temporal looming stimulus in the same retinal stream. Cursor distance, TTC and the observer threat score are telemetry only and never call an escape routine directly. A near-field disturbance value is also measured, but is intentionally not injected into MaleCNS until receptor-level mechanosensory mapping is validated.

A single DNp01/Giant Fiber spike is preserved as an immediate escape/take-off signal instead of being averaged away by a rate threshold. DNg02 MaleCNS subtypes provide sustained flight-thrust activity. DNg100 remains the primary forward-walking read-out, while bilateral DNa01/DNa02 activity contributes locomotor drive as well as steering. MDN provides backward drive.

Click the fly to open the native resizable control panel. It contains Overview, Brain, Care, Events, Perception and Life tabs. The Perception tab exposes temporal cursor/looming/TTC/optic-flow diagnostics. The persistent organism name can be changed only from the Life tab. Only the Windows title-bar close button is shown; closing the panel hides it without terminating the organism. The panel size and position persist across launches.

The current body is a 2.5-D kinematic motor decoder: x/y are desktop coordinates while altitude/vertical velocity are independent flight state. Grounded rendering uses an alternating tripod gait derived from locomotor telemetry. It is still not a complete musculoskeletal Drosophila simulation.

## Feeding

The Care tab can place a visible sugar drop anywhere on the desktop. Placement uses a temporary full-desktop click layer; the fly must physically contact the drop before it is consumed. Hunger, feeding count and last-feed time persist across launches.

Food is also rendered into the retinal panorama. Hunger is now represented as an internal homeostatic drive: it increases neural locomotor readiness and interacts with metabolic vitality, but it still does not directly steer the fly toward food. Exact sweet-receptor/gustatory-neuron injection is intentionally disabled until receptor-level identity is available in the bundled MaleCNS metadata.

## Validation

Run the deterministic tests:

```bash
python -m unittest discover -s tests -v
```

With the MaleCNS data available, validate both the early visual relay and the real motor groups/decoder:

```bash
python scripts/validate_graded_vision.py
python scripts/validate_motor_groups.py
```

The motor validation resolves real MaleCNS DNg02 subtypes, verifies a single DNp01 spike survives as take-off, checks DNa-driven locomotion and DNg02 flight thrust, and runs an end-to-end raw-retina challenge that must produce translational motor output.

## Roadmap

1. Neural foundation and reproducible Windows builds
2. Graded early visual-system modelling and validation
3. Identified MaleCNS descending-neuron output to a 2-D desktop body
4. Replace the kinematic body with a fuller biomechanical Drosophila body while preserving the closed loop
5. Olfactory, gustatory and mechanosensory world inputs
6. Mushroom-body learning where experimentally supportable
7. Persistent neural/plastic state across launches

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Scientific provenance

Flybit is a fork/derivative of **fly.ai** by alextitonis, released under the MIT License.

The MaleCNS v1.0 connectome was produced by FlyEM / HHMI Janelia and collaborators and is distributed separately under its own CC BY 4.0 terms. Flybit does not redistribute the raw connectome in this repository.

See [NOTICE.md](NOTICE.md) for attribution details.

## License

Code in this repository remains under the MIT License unless a file states otherwise. MaleCNS data remains under its original data license.
