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

The first Flybit foundation is intentionally conservative:

- loads the full MaleCNS network used by `fly.ai`
- 166,700 neurons
- 25,582,938 connectome connections
- uses the raw visual route into 6,006 photoreceptors
- shows live whole-network and descending-neuron activity
- does **not** bypass the eye with the upstream `FeatureDetectors` shortcut
- does **not** fake locomotion before a biological motor/body bridge exists
- builds a Windows executable through GitHub Actions

The upstream simulator is still a simplified neural model. MaleCNS is a measured wiring diagram, but the current dynamics do not yet reproduce every biological mechanism such as graded signalling, receptor-specific transmitter effects, neuromodulation or plasticity.

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

The first launch initializes the MaleCNS neural core. Move the mouse across the display and Flybit maps its horizontal position into the raw photoreceptor scene while showing neural activity.

## Roadmap

1. Neural foundation and reproducible Windows builds
2. Graded early visual-system modelling
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
