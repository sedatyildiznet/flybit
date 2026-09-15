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

## v0.1 foundation

The first executable deliberately stops before locomotion rather than faking it.

```
Windows cursor
    -> 1-D visual scene
    -> 6,006 MaleCNS photoreceptors
    -> fly.ai MaleCNS network
    -> descending-neuron activity
    -> live inspector
```

The upstream leaky-integrate-and-fire model is known to model the early graded visual system poorly. Flybit therefore does **not** use `FeatureDetectors` to bypass the eye in the desktop organism path.

## Roadmap

1. **Neural foundation** — reproducible MaleCNS load, raw sensory route, telemetry, Windows EXE.
2. **Graded vision** — biologically grounded graded early visual neurons so photoreceptor signals propagate without feature-injection shortcuts.
3. **Motor bridge** — map identified MaleCNS/VNC motor outputs to a Drosophila biomechanical body model without behaviour rules.
4. **Closed-loop desktop world** — rendered desktop environment feeds sensory organs and receives only physical body output.
5. **Feeding / mechanosensation** — add olfactory, gustatory and mechanosensory transduction from published biology.
6. **Plasticity** — mushroom-body dopamine-gated learning where experimentally supportable.
7. **Persistent individual** — persist neural/plastic state so one Flybit has a continuous history across launches.

## Scientific honesty

MaleCNS is a measured wiring diagram. The current simulator is **not** a complete biological brain emulation: membrane dynamics, receptor-specific transmitter effects, graded signalling, neuromodulation and plasticity are incomplete. Flybit should expose these limitations instead of hiding them behind scripted behaviour.
