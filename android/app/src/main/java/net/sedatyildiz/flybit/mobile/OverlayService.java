package net.sedatyildiz.flybit.mobile;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Intent;
import android.graphics.PixelFormat;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.provider.Settings;
import android.view.Gravity;
import android.view.WindowManager;

public final class OverlayService extends Service {
    public static final String ACTION_START = "net.sedatyildiz.flybit.action.START";
    public static final String ACTION_STOP = "net.sedatyildiz.flybit.action.STOP";
    public static final String ACTION_FEED = "net.sedatyildiz.flybit.action.FEED";
    public static final String ACTION_POKE = "net.sedatyildiz.flybit.action.POKE";

    private static final String CHANNEL_ID = "flybit_organism";
    private static final int NOTIFICATION_ID = 601;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private WindowManager windowManager;
    private FlySpriteView flyView;
    private SugarView sugarView;
    private WindowManager.LayoutParams flyParams;
    private WindowManager.LayoutParams sugarParams;
    private FlybitEngine engine;
    private long lastFrameNs;

    private final Runnable frameLoop = new Runnable() {
        @Override
        public void run() {
            if (engine == null || flyView == null || windowManager == null) {
                return;
            }

            long now = System.nanoTime();
            float dt = lastFrameNs == 0L ? 1f / 30f : (now - lastFrameNs) / 1_000_000_000f;
            lastFrameNs = now;
            engine.step(dt);

            FlybitEngine.Snapshot s = engine.snapshot();
            int width = getResources().getDisplayMetrics().widthPixels;
            int height = getResources().getDisplayMetrics().heightPixels;

            flyParams.x = Math.round(s.x * width - flyParams.width * 0.5f);
            flyParams.y = Math.round(s.y * height - flyParams.height * 0.5f);
            flyView.setMotion(s.heading, s.behavior);

            try {
                windowManager.updateViewLayout(flyView, flyParams);
                syncSugarOverlay(s, width, height);
            } catch (IllegalArgumentException ignored) {
                return;
            }

            handler.postDelayed(this, 33L);
        }
    };

    @Override
    public void onCreate() {
        super.onCreate();
        engine = new FlybitEngine(this);
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
        createNotificationChannel();
        startForeground(NOTIFICATION_ID, buildNotification());
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        String action = intent == null ? ACTION_START : intent.getAction();

        if (ACTION_STOP.equals(action)) {
            stopSelf();
            return START_NOT_STICKY;
        }

        if (ACTION_FEED.equals(action)) {
            engine.placeSugar();
        } else if (ACTION_POKE.equals(action)) {
            engine.registerTouchThreat();
        }

        if (Settings.canDrawOverlays(this)) {
            ensureOverlay();
        }

        return START_STICKY;
    }

    private void ensureOverlay() {
        if (flyView != null) {
            return;
        }

        int size = dp(82);
        flyView = new FlySpriteView(this);
        flyView.setOnClickListener(v -> {
            engine.registerTouchThreat();
            Intent open = new Intent(this, MainActivity.class);
            open.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_SINGLE_TOP);
            startActivity(open);
        });

        flyParams = new WindowManager.LayoutParams(
                size,
                size,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT
        );
        flyParams.gravity = Gravity.TOP | Gravity.START;

        FlybitEngine.Snapshot s = engine.snapshot();
        int width = getResources().getDisplayMetrics().widthPixels;
        int height = getResources().getDisplayMetrics().heightPixels;
        flyParams.x = Math.round(s.x * width - size * 0.5f);
        flyParams.y = Math.round(s.y * height - size * 0.5f);

        windowManager.addView(flyView, flyParams);
        lastFrameNs = System.nanoTime();
        handler.removeCallbacks(frameLoop);
        handler.post(frameLoop);
    }

    private void syncSugarOverlay(FlybitEngine.Snapshot s, int width, int height) {
        if (s.foodActive) {
            if (sugarView == null) {
                sugarView = new SugarView(this);
                int size = dp(30);
                sugarParams = new WindowManager.LayoutParams(
                        size,
                        size,
                        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                                | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                                | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                        PixelFormat.TRANSLUCENT
                );
                sugarParams.gravity = Gravity.TOP | Gravity.START;
                sugarParams.x = Math.round(s.foodX * width - size * 0.5f);
                sugarParams.y = Math.round(s.foodY * height - size * 0.5f);
                windowManager.addView(sugarView, sugarParams);
            } else {
                sugarParams.x = Math.round(s.foodX * width - sugarParams.width * 0.5f);
                sugarParams.y = Math.round(s.foodY * height - sugarParams.height * 0.5f);
                windowManager.updateViewLayout(sugarView, sugarParams);
            }
        } else if (sugarView != null) {
            try {
                windowManager.removeView(sugarView);
            } catch (IllegalArgumentException ignored) {
            }
            sugarView = null;
            sugarParams = null;
        }
    }

    private Notification buildNotification() {
        Intent open = new Intent(this, MainActivity.class);
        PendingIntent pending = PendingIntent.getActivity(
                this,
                0,
                open,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE
        );

        return new Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("Flybit")
                .setContentText("Mobile organism is living on your screen")
                .setSmallIcon(android.R.drawable.ic_menu_compass)
                .setContentIntent(pending)
                .setOngoing(true)
                .build();
    }

    private void createNotificationChannel() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                "Flybit organism",
                NotificationManager.IMPORTANCE_LOW
        );
        channel.setDescription("Keeps the screen organism alive while the overlay is enabled.");
        manager.createNotificationChannel(channel);
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    public void onDestroy() {
        handler.removeCallbacks(frameLoop);
        if (engine != null) {
            engine.persist();
        }
        if (windowManager != null) {
            if (flyView != null) {
                try {
                    windowManager.removeView(flyView);
                } catch (IllegalArgumentException ignored) {
                }
            }
            if (sugarView != null) {
                try {
                    windowManager.removeView(sugarView);
                } catch (IllegalArgumentException ignored) {
                }
            }
        }
        flyView = null;
        sugarView = null;
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}
