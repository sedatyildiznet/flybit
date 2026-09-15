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
- does **not** fake locomotion before a biological motor/body bridge exists
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

Move the mouse across the display. Its position is rendered as a dark target in the compound-eye scene. Flybit converts that scene to adapting photoreceptor contrast and passes it through the mixed graded/spiking MaleCNS network.

## Validation

Run the small deterministic tests:

```bash
python -m unittest discover -s tests -v
```

With the MaleCNS data available, validate the real early visual relay:

```bash
python scripts/validate_graded_vision.py
```

## Roadmap

1. Neural foundation and reproducible Windows builds
2. Graded early visual-system modelling and validation
3. MaleCNS/VNC motor output to biomechanical Drosophila body
4. Closed-loop desktop movement with no behaviour controller
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
