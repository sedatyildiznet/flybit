# Flybit Android

Flybit Android is the native mobile companion to the Windows desktop organism.

## What is preserved

The mobile loop follows the same separation used by desktop Flybit:

```text
world
→ sensory
→ modeled processing
→ physiology + memory
→ ethology
→ motor
→ body physics
→ world
```

Touch/contact, screen boundaries and the virtual sugar source are inputs to the organism. UI actions do not issue direct movement commands.

## Important evidence boundary

The Android APK does **not** execute the Python/Numba MaleCNS connectome. Android telemetry is explicitly labeled **MODELED MOBILE CNS**. The connectome-backed MaleCNS implementation remains in the desktop/Python target.

## Android behavior

- Native Android app, minSdk 26.
- Optional `SYSTEM_ALERT_WINDOW` overlay so Flybit can live above other apps.
- Foreground service keeps the organism alive while the overlay is enabled.
- Persistent identity, position, hunger, energy, sleep pressure, grooming need and threat memory.
- Bilateral odor model for sugar foraging.
- Physical food contact is required before feeding.
- Overview, Brain, Care and Logs tabs mirror the desktop control-panel concepts.
- Touching the overlay creates a contact/loom threat stimulus rather than a direct escape command.

## Build

A Gradle wrapper is intentionally not committed. CI installs Gradle and Android SDK explicitly.

```bash
gradle -p android clean assembleDebug
```

The GitHub release workflow publishes the installable debug-signed APK as `Flybit-Android.apk`. It is intended for direct GitHub testing, not Play Store distribution.
