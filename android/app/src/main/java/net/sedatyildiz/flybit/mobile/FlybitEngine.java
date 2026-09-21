package net.sedatyildiz.flybit.mobile;

import android.content.Context;
import android.content.SharedPreferences;

import java.time.Instant;
import java.time.ZonedDateTime;
import java.util.Locale;
import java.util.Random;

/**
 * Mobile Flybit organism loop.
 *
 * The Android port preserves the desktop architecture:
 * world -> sensory -> modeled processing -> physiology/memory -> ethology
 * -> motor -> body physics -> world.
 *
 * It intentionally does not pretend to execute the Python/Numba MaleCNS graph.
 */
public final class FlybitEngine {
    public static final String PREFS = "flybit_mobile_state";

    public static final class Snapshot {
        public final String displayName;
        public final String behavior;
        public final float x, y, heading;
        public final float hunger, energy, sleepPressure, groomingNeed;
        public final float arousal, threatMemory, odor;
        public final float forward, steer, escape, flight;
        public final boolean foodActive;
        public final float foodX, foodY;
        public final int feedings;
        public final String lastFeedAt;
        public final String lastEvent;

        Snapshot(
                String displayName,
                String behavior,
                float x,
                float y,
                float heading,
                float hunger,
                float energy,
                float sleepPressure,
                float groomingNeed,
                float arousal,
                float threatMemory,
                float odor,
                float forward,
                float steer,
                float escape,
                float flight,
                boolean foodActive,
                float foodX,
                float foodY,
                int feedings,
                String lastFeedAt,
                String lastEvent
        ) {
            this.displayName = displayName;
            this.behavior = behavior;
            this.x = x;
            this.y = y;
            this.heading = heading;
            this.hunger = hunger;
            this.energy = energy;
            this.sleepPressure = sleepPressure;
            this.groomingNeed = groomingNeed;
            this.arousal = arousal;
            this.threatMemory = threatMemory;
            this.odor = odor;
            this.forward = forward;
            this.steer = steer;
            this.escape = escape;
            this.flight = flight;
            this.foodActive = foodActive;
            this.foodX = foodX;
            this.foodY = foodY;
            this.feedings = feedings;
            this.lastFeedAt = lastFeedAt;
            this.lastEvent = lastEvent;
        }
    }

    private final SharedPreferences prefs;
    private final Random random;

    private String displayName;
    private String behavior = "REST";
    private String lastEvent = "mobile organism initialized";
    private String lastFeedAt = "";

    private float x;
    private float y;
    private float heading;
    private float hunger;
    private float energy;
    private float sleepPressure;
    private float groomingNeed;
    private float arousal;
    private float threatMemory;

    private float touchThreat;
    private float odor;
    private float previousOdor;
    private float forward;
    private float steer;
    private float escape;
    private float flight;

    private boolean foodActive;
    private float foodX;
    private float foodY;
    private int feedings;

    private float persistAccumulator;
    private float eventAccumulator;

    public FlybitEngine(Context context) {
        prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        long seed = prefs.getLong("seed", 0L);
        if (seed == 0L) {
            seed = System.nanoTime() ^ Instant.now().toEpochMilli();
            prefs.edit().putLong("seed", seed).apply();
        }
        random = new Random(seed ^ System.nanoTime());

        displayName = normalizeName(prefs.getString("display_name", "Flybit"));
        x = prefs.getFloat("x", 0.55f);
        y = prefs.getFloat("y", 0.42f);
        heading = prefs.getFloat("heading", random.nextFloat() * 6.2831855f);
        hunger = prefs.getFloat("hunger", 0.35f);
        energy = prefs.getFloat("energy", 0.82f);
        sleepPressure = prefs.getFloat("sleep_pressure", 0.35f);
        groomingNeed = prefs.getFloat("grooming_need", 0.18f);
        threatMemory = prefs.getFloat("threat_memory", 0f);
        foodActive = prefs.getBoolean("food_active", false);
        foodX = prefs.getFloat("food_x", 0.25f);
        foodY = prefs.getFloat("food_y", 0.72f);
        feedings = prefs.getInt("feedings", 0);
        lastFeedAt = prefs.getString("last_feed_at", "");
        lastEvent = prefs.getString("last_event", lastEvent);

        long previous = prefs.getLong("last_simulated_ms", System.currentTimeMillis());
        float offlineSeconds = Math.min(6f * 3600f, Math.max(0f,
                (System.currentTimeMillis() - previous) / 1000f));
        applyOfflinePhysiology(offlineSeconds);
        persist();
    }

    public synchronized void step(float dt) {
        dt = clamp(dt, 0.001f, 0.1f);
        eventAccumulator += dt;

        // physiology
        hunger = clamp01(hunger + dt * 0.00055f);
        groomingNeed = clamp01(groomingNeed + dt * 0.00017f);
        threatMemory = clamp01(threatMemory - dt * 0.010f);
        touchThreat = Math.max(0f, touchThreat - dt * 1.45f);

        int hour = ZonedDateTime.now().getHour();
        boolean circadianNight = hour >= 23 || hour < 7;
        sleepPressure = clamp01(sleepPressure + dt * (circadianNight ? 0.0012f : -0.00035f));

        // sensory: bilateral olfaction, screen-edge cues, contact/loom pulse.
        float leftOdor = foodActive ? odorAtAntenna(-1f) : 0f;
        float rightOdor = foodActive ? odorAtAntenna(1f) : 0f;
        odor = 0.5f * (leftOdor + rightOdor);
        float odorDelta = odor - previousOdor;
        previousOdor = odor;

        float edgeLeft = ramp(0.09f - x, 0.09f);
        float edgeRight = ramp(x - 0.91f, 0.09f);
        float edgeTop = ramp(0.08f - y, 0.08f);
        float edgeBottom = ramp(y - 0.92f, 0.08f);
        float boundaryThreat = Math.max(Math.max(edgeLeft, edgeRight), Math.max(edgeTop, edgeBottom));

        // modeled neural/arousal integration
        float threatInput = Math.max(touchThreat, boundaryThreat * 0.35f);
        arousal += (threatInput - arousal) * Math.min(1f, dt * 7.0f);
        threatMemory = clamp01(Math.max(threatMemory, arousal * 0.82f));

        // ethology selection: internal state selects behavior, not direct UI commands.
        if (arousal > 0.56f) {
            behavior = "ESCAPE";
        } else if (sleepPressure > 0.80f && energy < 0.72f) {
            behavior = "REST";
        } else if (groomingNeed > 0.78f && hunger < 0.78f) {
            behavior = "GROOM";
        } else if (foodActive && hunger > 0.46f && odor > 0.13f) {
            behavior = "FORAGE";
        } else if (energy > 0.32f) {
            behavior = random.nextFloat() < dt * 0.055f ? "FLIGHT" : "WALK";
        } else {
            behavior = "REST";
        }

        // motor decoding
        float turnNoise = (random.nextFloat() - 0.5f) * 0.72f;
        forward = 0f;
        steer = 0f;
        escape = 0f;
        flight = 0f;

        switch (behavior) {
            case "ESCAPE":
                escape = clamp01(0.68f + arousal * 0.32f);
                forward = 0.75f + 0.25f * escape;
                steer = turnNoise * 1.8f;
                energy = clamp01(energy - dt * 0.018f);
                break;
            case "FORAGE":
                forward = 0.28f + hunger * 0.34f;
                steer = clamp((rightOdor - leftOdor) * 3.4f + odorDelta * 0.7f + turnNoise * 0.15f, -1f, 1f);
                energy = clamp01(energy - dt * 0.0045f);
                break;
            case "FLIGHT":
                flight = 0.58f + random.nextFloat() * 0.25f;
                forward = 0.72f;
                steer = turnNoise;
                energy = clamp01(energy - dt * 0.013f);
                break;
            case "WALK":
                forward = 0.18f + random.nextFloat() * 0.25f;
                steer = turnNoise * 0.45f;
                energy = clamp01(energy - dt * 0.003f);
                break;
            case "GROOM":
                groomingNeed = clamp01(groomingNeed - dt * 0.19f);
                energy = clamp01(energy + dt * 0.002f);
                steer = turnNoise * 0.05f;
                break;
            default:
                energy = clamp01(energy + dt * 0.010f);
                sleepPressure = clamp01(sleepPressure - dt * 0.009f);
                break;
        }

        // boundary cue contributes sensory steering instead of teleporting the fly.
        float awayX = edgeLeft - edgeRight;
        float awayY = edgeTop - edgeBottom;
        if (boundaryThreat > 0f) {
            float desired = (float) Math.atan2(awayY, awayX);
            steer += angleDelta(heading, desired) * 0.45f;
        }

        heading = wrapAngle(heading + steer * dt * 3.8f);
        float speed = forward * (behavior.equals("FLIGHT") ? 0.22f : 0.095f);
        x += (float) Math.cos(heading) * speed * dt;
        y += (float) Math.sin(heading) * speed * dt;

        // physical collision with screen boundary.
        if (x < 0.015f || x > 0.985f) {
            x = clamp(x, 0.015f, 0.985f);
            heading = wrapAngle((float) Math.PI - heading);
        }
        if (y < 0.02f || y > 0.98f) {
            y = clamp(y, 0.02f, 0.98f);
            heading = wrapAngle(-heading);
        }

        // food is only consumed after physical contact.
        if (foodActive) {
            float dx = foodX - x;
            float dy = foodY - y;
            if (dx * dx + dy * dy < 0.0011f) {
                consumeFood();
            }
        }

        persistAccumulator += dt;
        if (persistAccumulator >= 1.0f) {
            persistAccumulator = 0f;
            persist();
        }
    }

    public synchronized void registerTouchThreat() {
        touchThreat = 1f;
        threatMemory = Math.max(threatMemory, 0.52f);
        setEvent("contact/loom stimulus detected");
    }

    public synchronized void placeSugar() {
        // Source is placed in world space; movement is still odor driven.
        float distance;
        do {
            foodX = 0.10f + random.nextFloat() * 0.80f;
            foodY = 0.12f + random.nextFloat() * 0.76f;
            float dx = foodX - x;
            float dy = foodY - y;
            distance = dx * dx + dy * dy;
        } while (distance < 0.05f);
        foodActive = true;
        setEvent("sugar source placed; odor field active");
        persist();
    }

    public synchronized void setDisplayName(String name) {
        displayName = normalizeName(name);
        setEvent("identity updated");
        persist();
    }

    public synchronized void persist() {
        Snapshot s = snapshot();
        prefs.edit()
                .putString("display_name", s.displayName)
                .putFloat("x", s.x)
                .putFloat("y", s.y)
                .putFloat("heading", s.heading)
                .putFloat("hunger", s.hunger)
                .putFloat("energy", s.energy)
                .putFloat("sleep_pressure", s.sleepPressure)
                .putFloat("grooming_need", s.groomingNeed)
                .putFloat("threat_memory", s.threatMemory)
                .putFloat("arousal", s.arousal)
                .putFloat("odor", s.odor)
                .putFloat("motor_forward", s.forward)
                .putFloat("motor_steer", s.steer)
                .putFloat("motor_escape", s.escape)
                .putFloat("motor_flight", s.flight)
                .putString("behavior", s.behavior)
                .putBoolean("food_active", s.foodActive)
                .putFloat("food_x", s.foodX)
                .putFloat("food_y", s.foodY)
                .putInt("feedings", s.feedings)
                .putString("last_feed_at", s.lastFeedAt)
                .putString("last_event", s.lastEvent)
                .putLong("last_simulated_ms", System.currentTimeMillis())
                .apply();
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(
                displayName, behavior, x, y, heading,
                hunger, energy, sleepPressure, groomingNeed,
                arousal, threatMemory, odor,
                forward, steer, escape, flight,
                foodActive, foodX, foodY, feedings, lastFeedAt, lastEvent
        );
    }

    private void consumeFood() {
        foodActive = false;
        hunger = clamp01(hunger - 0.47f);
        energy = clamp01(energy + 0.22f);
        feedings += 1;
        lastFeedAt = Instant.now().toString();
        setEvent("sugar contact: feeding completed");
    }

    private float odorAtAntenna(float side) {
        float antennaForward = 0.018f;
        float antennaSide = 0.012f * side;
        float cos = (float) Math.cos(heading);
        float sin = (float) Math.sin(heading);
        float ax = x + cos * antennaForward - sin * antennaSide;
        float ay = y + sin * antennaForward + cos * antennaSide;
        float dx = foodX - ax;
        float dy = foodY - ay;
        float d2 = dx * dx + dy * dy;
        return clamp01((float) Math.exp(-d2 / 0.055f));
    }

    private void applyOfflinePhysiology(float seconds) {
        if (seconds <= 0f) return;
        hunger = clamp01(hunger + seconds * 0.00010f);
        energy = clamp01(energy + seconds * 0.00008f);
        threatMemory = clamp01(threatMemory - seconds * 0.00015f);
    }

    private void setEvent(String event) {
        if (eventAccumulator < 0.08f && event.equals(lastEvent)) return;
        eventAccumulator = 0f;
        lastEvent = String.format(Locale.US, "%s · %s", Instant.now().toString(), event);
    }

    private static String normalizeName(String raw) {
        String text = raw == null ? "" : raw.trim().replaceAll("\\s+", " ");
        if (text.isEmpty()) return "Flybit";
        return text.substring(0, Math.min(32, text.length()));
    }

    private static float ramp(float v, float width) {
        return clamp01(v / Math.max(0.0001f, width));
    }

    private static float angleDelta(float from, float to) {
        float d = wrapAngle(to - from);
        if (d > Math.PI) d -= (float) (Math.PI * 2.0);
        return d;
    }

    private static float wrapAngle(float angle) {
        float twoPi = (float) (Math.PI * 2.0);
        while (angle < 0f) angle += twoPi;
        while (angle >= twoPi) angle -= twoPi;
        return angle;
    }

    private static float clamp01(float value) {
        return clamp(value, 0f, 1f);
    }

    private static float clamp(float value, float low, float high) {
        return Math.max(low, Math.min(high, value));
    }
}
