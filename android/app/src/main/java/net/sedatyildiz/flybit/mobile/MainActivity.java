package net.sedatyildiz.flybit.mobile;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.Locale;
import java.util.Random;

public final class MainActivity extends Activity {
    private static final int BG = 0xFF0B0D10;
    private static final int CARD = 0xFF15191F;
    private static final int TEXT = 0xFFF2F4F7;
    private static final int MUTED = 0xFF9AA4B2;
    private static final int ACCENT = 0xFF67E8F9;
    private static final String STATE_ACTIVE_TAB = "active_tab";
    private static final String PREF_PENDING_OVERLAY_ACTION = "pending_overlay_action";
    private static final long OVERLAY_HEARTBEAT_TIMEOUT_MS = 3_000L;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private SharedPreferences prefs;
    private FrameLayout content;
    private int activeTab = 0;

    private TextView status;
    private TextView behavior;
    private TextView motor;
    private TextView sensory;
    private TextView physiology;
    private TextView careDetail;
    private TextView logs;
    private ProgressBar hunger;
    private ProgressBar energy;
    private ProgressBar sleep;
    private EditText nameInput;
    private HabitatView habitat;
    private BrainMapView brainMap;

    private final Runnable refresh = new Runnable() {
        @Override
        public void run() {
            updateTelemetry();
            handler.postDelayed(this, 450L);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences(FlybitEngine.PREFS, MODE_PRIVATE);
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);

        LinearLayout root = vertical();
        root.setPadding(dp(16), dp(14), dp(16), dp(16));
        root.setBackgroundColor(BG);

        LinearLayout header = horizontal();
        LinearLayout titleBox = vertical();
        TextView title = text("FLYBIT", 24, TEXT);
        title.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        titleBox.addView(title);
        titleBox.addView(text("Android screen organism · v" + BuildConfig.VERSION_NAME, 12, MUTED));
        header.addView(titleBox, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        status = text("● OFFLINE", 12, ACCENT);
        status.setGravity(Gravity.END);
        header.addView(status);
        root.addView(header);

        TextView evidence = text(
                "MODELED MOBILE CNS · world → sensory → processing → physiology/memory → ethology → motor → body",
                11,
                MUTED
        );
        evidence.setPadding(0, dp(8), 0, dp(10));
        root.addView(evidence);

        HorizontalScrollView tabScroll = new HorizontalScrollView(this);
        tabScroll.setHorizontalScrollBarEnabled(false);
        LinearLayout tabs = horizontal();
        String[] labels = {"Overview", "Brain", "Care", "Logs"};
        for (int i = 0; i < labels.length; i++) {
            final int index = i;
            Button button = new Button(this);
            button.setText(labels[i]);
            button.setAllCaps(false);
            button.setTextColor(TEXT);
            button.setBackgroundColor(CARD);
            button.setOnClickListener(v -> showTab(index));
            LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(dp(108), dp(46));
            bp.setMargins(0, 0, dp(8), 0);
            tabs.addView(button, bp);
        }
        tabScroll.addView(tabs);
        root.addView(tabScroll);

        content = new FrameLayout(this);
        LinearLayout.LayoutParams cp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f
        );
        cp.topMargin = dp(12);
        root.addView(content, cp);

        setContentView(root);
        activeTab = savedInstanceState == null ? 0 : savedInstanceState.getInt(STATE_ACTIVE_TAB, 0);
        showTab(activeTab);
    }

    @Override
    protected void onResume() {
        super.onResume();
        handler.removeCallbacks(refresh);
        handler.post(refresh);

        String pendingAction = prefs.getString(PREF_PENDING_OVERLAY_ACTION, "");
        if (!pendingAction.isEmpty() && Settings.canDrawOverlays(this)) {
            prefs.edit().remove(PREF_PENDING_OVERLAY_ACTION).apply();
            handler.post(() -> sendServiceAction(pendingAction));
        }
    }

    @Override
    protected void onPause() {
        handler.removeCallbacks(refresh);
        super.onPause();
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        outState.putInt(STATE_ACTIVE_TAB, activeTab);
        super.onSaveInstanceState(outState);
    }

    private void showTab(int index) {
        activeTab = index;
        clearTabReferences();
        content.removeAllViews();

        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        LinearLayout body = vertical();
        body.setPadding(0, 0, 0, dp(24));
        scroll.addView(body);

        if (index == 0) buildOverview(body);
        else if (index == 1) buildBrain(body);
        else if (index == 2) buildCare(body);
        else buildLogs(body);

        content.addView(scroll);
        updateTelemetry();
    }

    private void buildOverview(LinearLayout body) {
        habitat = new HabitatView();
        body.addView(habitat, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(220)
        ));

        body.addView(section("LIVE STATE"));
        behavior = metricCard(body, "BEHAVIOR");
        physiology = metricCard(body, "PHYSIOLOGY");
        sensory = metricCard(body, "SENSORY");
        motor = metricCard(body, "DESCENDING MOTOR OUTPUT");

        TextView note = text(
                "Android uses a native modeled organism loop. It does not claim to run the Python/Numba MaleCNS connectome; evidence class is kept explicit.",
                12,
                MUTED
        );
        note.setPadding(dp(4), dp(14), dp(4), 0);
        body.addView(note);
    }

    private void buildBrain(LinearLayout body) {
        TextView title = section("MODELED MOBILE CNS MAP");
        body.addView(title);

        brainMap = new BrainMapView();
        body.addView(brainMap, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(360)
        ));

        TextView note = text(
                "Activity points visualize the Android sensory/arousal/motor model. They are not MaleCNS neuron IDs. Desktop Flybit remains the connectome-backed implementation.",
                12,
                MUTED
        );
        note.setPadding(dp(4), dp(12), dp(4), 0);
        body.addView(note);

        sensory = metricCard(body, "SENSORY CHANNELS");
        motor = metricCard(body, "MOTOR DECODER");
    }

    private void buildCare(LinearLayout body) {
        body.addView(section("CARE & FEEDING"));

        nameInput = new EditText(this);
        nameInput.setSingleLine(true);
        nameInput.setTextColor(TEXT);
        nameInput.setHintTextColor(MUTED);
        nameInput.setHint("Flybit name");
        nameInput.setText(prefs.getString("display_name", "Flybit"));
        nameInput.setBackgroundColor(CARD);
        nameInput.setPadding(dp(14), dp(12), dp(14), dp(12));
        body.addView(nameInput, matchWrap(dp(6)));

        Button saveName = actionButton("Save name");
        saveName.setOnClickListener(v -> saveDisplayName());
        body.addView(saveName, matchWrap(dp(6)));

        body.addView(label("HUNGER"));
        hunger = progress();
        body.addView(hunger);

        body.addView(label("ENERGY"));
        energy = progress();
        body.addView(energy);

        body.addView(label("SLEEP PRESSURE"));
        sleep = progress();
        body.addView(sleep);

        careDetail = metricCard(body, "CARE STATE");

        Button feed = actionButton("Place sugar");
        feed.setOnClickListener(v -> sendServiceAction(OverlayService.ACTION_FEED));
        body.addView(feed, matchWrap(dp(6)));

        Button start = actionButton("Start screen organism");
        start.setOnClickListener(v -> startOverlay());
        body.addView(start, matchWrap(dp(6)));

        Button stop = actionButton("Stop screen organism");
        stop.setOnClickListener(v -> stopService(new Intent(this, OverlayService.class)));
        body.addView(stop, matchWrap(dp(6)));

        Button permission = actionButton("Overlay permission");
        permission.setOnClickListener(v -> {
            Intent intent = new Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:" + getPackageName())
            );
            startActivity(intent);
        });
        body.addView(permission, matchWrap(dp(6)));

        TextView note = text(
                "Sugar is a world-space source. Flybit receives a bilateral odor gradient and only feeds after physical contact; the feed button does not issue a move-to command.",
                12,
                MUTED
        );
        note.setPadding(dp(4), dp(12), dp(4), 0);
        body.addView(note);
    }

    private void buildLogs(LinearLayout body) {
        body.addView(section("ORGANISM LOG"));
        logs = text("—", 12, TEXT);
        logs.setBackgroundColor(CARD);
        logs.setPadding(dp(14), dp(14), dp(14), dp(14));
        logs.setTextIsSelectable(true);
        body.addView(logs, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(320)
        ));

        TextView architecture = text(
                "Evidence discipline\nMEASURED: Android touch/contact and screen boundary geometry.\nMODELED: olfaction, arousal, physiology, ethology and motor decoding.\nSYNTHETIC: sugar source and virtual body/world.",
                12,
                MUTED
        );
        architecture.setPadding(dp(4), dp(14), dp(4), 0);
        body.addView(architecture);
    }

    private void startOverlay() {
        sendServiceAction(OverlayService.ACTION_START);
    }

    private void requestOverlayPermissionThen(String action) {
        prefs.edit().putString(PREF_PENDING_OVERLAY_ACTION, action).apply();
        Intent intent = new Intent(
                Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:" + getPackageName())
        );
        startActivity(intent);
    }

    private void sendServiceAction(String action) {
        boolean requiresOverlay = OverlayService.ACTION_START.equals(action)
                || OverlayService.ACTION_FEED.equals(action);
        if (requiresOverlay && !Settings.canDrawOverlays(this)) {
            requestOverlayPermissionThen(action);
            return;
        }

        Intent intent = new Intent(this, OverlayService.class);
        intent.setAction(action);
        startForegroundService(intent);
    }

    private void saveDisplayName() {
        String normalized = FlybitEngine.normalizeDisplayName(nameInput.getText().toString());
        nameInput.setText(normalized);

        if (isOverlayAlive()) {
            Intent intent = new Intent(this, OverlayService.class);
            intent.setAction(OverlayService.ACTION_SET_NAME);
            intent.putExtra(OverlayService.EXTRA_DISPLAY_NAME, normalized);
            startForegroundService(intent);
        } else {
            FlybitEngine.storeDisplayName(this, normalized);
        }
        updateTelemetry();
    }

    private boolean isOverlayAlive() {
        boolean running = prefs.getBoolean(OverlayService.PREF_OVERLAY_RUNNING, false);
        long heartbeat = prefs.getLong(OverlayService.PREF_OVERLAY_HEARTBEAT_MS, 0L);
        long age = heartbeat <= 0L ? Long.MAX_VALUE : Math.max(0L, System.currentTimeMillis() - heartbeat);
        boolean alive = running
                && age <= OVERLAY_HEARTBEAT_TIMEOUT_MS
                && Settings.canDrawOverlays(this);

        if (running && !alive) {
            prefs.edit()
                    .putBoolean(OverlayService.PREF_OVERLAY_RUNNING, false)
                    .putLong(OverlayService.PREF_OVERLAY_HEARTBEAT_MS, 0L)
                    .apply();
        }
        return alive;
    }

    private void updateTelemetry() {
        if (prefs == null) return;

        boolean overlay = isOverlayAlive();
        if (status != null) {
            status.setText(overlay ? "● LIVING ON SCREEN" : "● PANEL MODE");
        }

        String b = prefs.getString("behavior", "REST");
        float h = prefs.getFloat("hunger", 0.35f);
        float e = prefs.getFloat("energy", 0.82f);
        float sp = prefs.getFloat("sleep_pressure", 0.35f);
        float groom = prefs.getFloat("grooming_need", 0.18f);
        float threat = prefs.getFloat("threat_memory", 0f);
        float arousal = prefs.getFloat("arousal", 0f);
        float odor = prefs.getFloat("odor", 0f);
        float fwd = prefs.getFloat("motor_forward", 0f);
        float steerV = prefs.getFloat("motor_steer", 0f);
        float escapeV = prefs.getFloat("motor_escape", 0f);
        float flightV = prefs.getFloat("motor_flight", 0f);

        if (behavior != null) {
            behavior.setText("BEHAVIOR\n" + b + "  ·  overlay " + (overlay ? "active" : "stopped"));
        }
        if (physiology != null) {
            physiology.setText(String.format(
                    Locale.US,
                    "PHYSIOLOGY\nhunger %.0f%%  ·  energy %.0f%%  ·  sleep %.0f%%  ·  groom %.0f%%",
                    h * 100f, e * 100f, sp * 100f, groom * 100f
            ));
        }
        if (sensory != null) {
            sensory.setText(String.format(
                    Locale.US,
                    "SENSORY\nodor %.3f  ·  arousal %.3f  ·  threat memory %.3f",
                    odor, arousal, threat
            ));
        }
        if (motor != null) {
            motor.setText(String.format(
                    Locale.US,
                    "DESCENDING MOTOR OUTPUT\nforward %.2f  ·  steer %+.2f  ·  escape %.2f  ·  flight %.2f",
                    fwd, steerV, escapeV, flightV
            ));
        }
        if (hunger != null) hunger.setProgress(Math.round(h * 100f));
        if (energy != null) energy.setProgress(Math.round(e * 100f));
        if (sleep != null) sleep.setProgress(Math.round(sp * 100f));

        if (careDetail != null) {
            int feedings = prefs.getInt("feedings", 0);
            boolean food = prefs.getBoolean("food_active", false);
            String lastFeed = prefs.getString("last_feed_at", "");
            careDetail.setText(
                    "CARE STATE\nfeedings " + feedings
                            + "  ·  sugar " + (food ? "active" : "none")
                            + "\nlast feed " + (lastFeed == null || lastFeed.isEmpty() ? "never" : lastFeed)
            );
        }

        if (logs != null) {
            String event = prefs.getString("last_event", "No organism event recorded yet.");
            long simulated = prefs.getLong("last_simulated_ms", 0L);
            long heartbeat = prefs.getLong(OverlayService.PREF_OVERLAY_HEARTBEAT_MS, 0L);
            long heartbeatAge = heartbeat <= 0L
                    ? -1L
                    : Math.max(0L, System.currentTimeMillis() - heartbeat);
            logs.setText(
                    event
                            + "\n\nlast simulation ms: " + simulated
                            + "\nmode: MODELED MOBILE CNS"
                            + "\noverlay permission: " + Settings.canDrawOverlays(this)
                            + "\noverlay heartbeat age ms: " + heartbeatAge
            );
        }

        if (habitat != null) habitat.invalidate();
        if (brainMap != null) brainMap.invalidate();
    }

    private void clearTabReferences() {
        behavior = null;
        motor = null;
        sensory = null;
        physiology = null;
        careDetail = null;
        logs = null;
        hunger = null;
        energy = null;
        sleep = null;
        nameInput = null;
        habitat = null;
        brainMap = null;
    }

    private TextView metricCard(LinearLayout parent, String title) {
        TextView card = text(title + "\n—", 13, TEXT);
        card.setBackgroundColor(CARD);
        card.setPadding(dp(14), dp(14), dp(14), dp(14));
        LinearLayout.LayoutParams lp = matchWrap(dp(7));
        parent.addView(card, lp);
        return card;
    }

    private TextView section(String value) {
        TextView t = text(value, 12, MUTED);
        t.setTypeface(android.graphics.Typeface.DEFAULT_BOLD);
        t.setPadding(dp(4), dp(14), dp(4), dp(8));
        return t;
    }

    private TextView label(String value) {
        TextView t = text(value, 11, MUTED);
        t.setPadding(dp(4), dp(12), dp(4), dp(4));
        return t;
    }

    private ProgressBar progress() {
        ProgressBar p = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        p.setMax(100);
        return p;
    }

    private Button actionButton(String value) {
        Button b = new Button(this);
        b.setText(value);
        b.setAllCaps(false);
        b.setTextColor(TEXT);
        b.setBackgroundColor(CARD);
        return b;
    }

    private LinearLayout.LayoutParams matchWrap(int bottom) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        lp.bottomMargin = bottom;
        return lp;
    }

    private LinearLayout vertical() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.VERTICAL);
        return l;
    }

    private LinearLayout horizontal() {
        LinearLayout l = new LinearLayout(this);
        l.setOrientation(LinearLayout.HORIZONTAL);
        l.setGravity(Gravity.CENTER_VERTICAL);
        return l;
    }

    private TextView text(String value, int sp, int color) {
        TextView t = new TextView(this);
        t.setText(value);
        t.setTextSize(sp);
        t.setTextColor(color);
        return t;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private final class HabitatView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

        HabitatView() {
            super(MainActivity.this);
            setBackgroundColor(CARD);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float x = prefs.getFloat("x", 0.55f) * getWidth();
            float y = prefs.getFloat("y", 0.42f) * getHeight();
            float heading = prefs.getFloat("heading", 0f);
            boolean food = prefs.getBoolean("food_active", false);

            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(1));
            paint.setColor(0x334A5568);
            for (int i = 1; i < 5; i++) {
                canvas.drawLine(getWidth() * i / 5f, 0, getWidth() * i / 5f, getHeight(), paint);
            }

            if (food) {
                paint.setStyle(Paint.Style.FILL);
                paint.setColor(0xFFE9E5D1);
                canvas.drawCircle(
                        prefs.getFloat("food_x", 0.25f) * getWidth(),
                        prefs.getFloat("food_y", 0.72f) * getHeight(),
                        dp(5),
                        paint
                );
            }

            canvas.save();
            canvas.rotate((float) Math.toDegrees(heading), x, y);
            paint.setColor(0xFF090909);
            paint.setStyle(Paint.Style.FILL);
            canvas.drawOval(x - dp(5), y - dp(10), x + dp(5), y + dp(10), paint);
            paint.setColor(0x775A6675);
            Path wing = new Path();
            wing.moveTo(x, y - dp(3));
            wing.lineTo(x - dp(16), y - dp(10));
            wing.lineTo(x - dp(10), y + dp(4));
            wing.close();
            canvas.drawPath(wing, paint);
            Path wing2 = new Path();
            wing2.moveTo(x, y - dp(3));
            wing2.lineTo(x + dp(16), y - dp(10));
            wing2.lineTo(x + dp(10), y + dp(4));
            wing2.close();
            canvas.drawPath(wing2, paint);
            canvas.restore();

            paint.setColor(MUTED);
            paint.setTextSize(dp(11));
            canvas.drawText("LIVE WORLD PREVIEW", dp(12), dp(20), paint);
        }
    }

    private final class BrainMapView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Random random = new Random(600L);
        private final float[] px = new float[96];
        private final float[] py = new float[96];

        BrainMapView() {
            super(MainActivity.this);
            setBackgroundColor(CARD);
            for (int i = 0; i < px.length; i++) {
                px[i] = 0.08f + random.nextFloat() * 0.84f;
                py[i] = 0.10f + random.nextFloat() * 0.80f;
            }
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float arousal = prefs.getFloat("arousal", 0f);
            float odor = prefs.getFloat("odor", 0f);
            float fwd = prefs.getFloat("motor_forward", 0f);
            float esc = prefs.getFloat("motor_escape", 0f);
            float activity = Math.min(1f, 0.08f + arousal * 0.38f + odor * 0.24f + fwd * 0.18f + esc * 0.38f);

            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(1));
            paint.setColor(0x223A4554);
            canvas.drawOval(dp(30), dp(45), getWidth() - dp(30), getHeight() - dp(38), paint);

            for (int i = 0; i < px.length; i++) {
                float x = px[i] * getWidth();
                float y = py[i] * getHeight();
                float phase = ((i * 37) % 100) / 100f;
                boolean active = phase < activity;
                paint.setStyle(Paint.Style.FILL);
                paint.setColor(active ? ACCENT : 0xFF46505E);
                canvas.drawCircle(x, y, active ? dp(2.3f) : dp(1.4f), paint);
            }

            paint.setColor(TEXT);
            paint.setTextSize(dp(11));
            canvas.drawText("MODELED · MOBILE CNS", dp(12), dp(20), paint);
            paint.setColor(MUTED);
            canvas.drawText(
                    String.format(Locale.US, "activity %.0f%%", activity * 100f),
                    dp(12),
                    getHeight() - dp(12),
                    paint
            );
        }
    }

    private int dp(float value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
