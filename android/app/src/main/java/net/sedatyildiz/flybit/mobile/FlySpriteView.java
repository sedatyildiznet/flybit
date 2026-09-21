package net.sedatyildiz.flybit.mobile;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.View;

public final class FlySpriteView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private float heading;
    private String behavior = "REST";

    public FlySpriteView(Context context) {
        super(context);
        setBackgroundColor(0x00000000);
        setClickable(true);
    }

    public void setMotion(float heading, String behavior) {
        this.heading = heading;
        this.behavior = behavior == null ? "REST" : behavior;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cx = getWidth() * 0.5f;
        float cy = getHeight() * 0.5f;
        float scale = Math.min(getWidth(), getHeight()) / 82f;

        canvas.save();
        canvas.rotate((float) Math.toDegrees(heading), cx, cy);

        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.4f * scale);
        paint.setStrokeCap(Paint.Cap.ROUND);
        paint.setColor(0xCC171717);

        for (int i = -1; i <= 1; i++) {
            float y = cy + i * 5f * scale;
            canvas.drawLine(cx - 5f * scale, y, cx - 17f * scale, y - 9f * scale, paint);
            canvas.drawLine(cx - 17f * scale, y - 9f * scale, cx - 23f * scale, y - 5f * scale, paint);
            canvas.drawLine(cx + 5f * scale, y, cx + 17f * scale, y + 9f * scale, paint);
            canvas.drawLine(cx + 17f * scale, y + 9f * scale, cx + 23f * scale, y + 5f * scale, paint);
        }

        paint.setStyle(Paint.Style.FILL);
        paint.setColor(0x553E4652);
        Path leftWing = new Path();
        leftWing.moveTo(cx - 1f * scale, cy - 3f * scale);
        leftWing.cubicTo(
                cx - 8f * scale, cy - 20f * scale,
                cx - 25f * scale, cy - 18f * scale,
                cx - 18f * scale, cy + 2f * scale
        );
        leftWing.close();
        canvas.drawPath(leftWing, paint);

        Path rightWing = new Path();
        rightWing.moveTo(cx + 1f * scale, cy - 3f * scale);
        rightWing.cubicTo(
                cx + 8f * scale, cy - 20f * scale,
                cx + 25f * scale, cy - 18f * scale,
                cx + 18f * scale, cy + 2f * scale
        );
        rightWing.close();
        canvas.drawPath(rightWing, paint);

        paint.setColor(0xFF151515);
        canvas.drawOval(
                cx - 6.5f * scale, cy - 12f * scale,
                cx + 6.5f * scale, cy + 11f * scale,
                paint
        );

        paint.setColor(0xFF2A2A2A);
        canvas.drawCircle(cx, cy - 13f * scale, 6.4f * scale, paint);

        paint.setColor(0xFF5C1515);
        canvas.drawCircle(cx - 3.6f * scale, cy - 14f * scale, 3.0f * scale, paint);
        canvas.drawCircle(cx + 3.6f * scale, cy - 14f * scale, 3.0f * scale, paint);

        paint.setColor(0xAA7A7F85);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(0.8f * scale);
        for (int i = 0; i < 5; i++) {
            float yy = cy - 1f * scale + i * 3f * scale;
            canvas.drawLine(cx - 5.2f * scale, yy, cx + 5.2f * scale, yy, paint);
        }

        if ("FLIGHT".equals(behavior) || "ESCAPE".equals(behavior)) {
            paint.setColor(0x225A6A80);
            paint.setStrokeWidth(2.4f * scale);
            canvas.drawLine(cx - 5f * scale, cy - 6f * scale, cx - 28f * scale, cy - 22f * scale, paint);
            canvas.drawLine(cx + 5f * scale, cy - 6f * scale, cx + 28f * scale, cy - 22f * scale, paint);
        }

        canvas.restore();
    }
}

final class SugarView extends View {
    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);

    SugarView(Context context) {
        super(context);
        setBackgroundColor(0x00000000);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        float cx = getWidth() * 0.5f;
        float cy = getHeight() * 0.5f;
        paint.setStyle(Paint.Style.FILL);
        paint.setColor(0xD9F4F1DE);
        canvas.drawCircle(cx, cy, Math.min(cx, cy) * 0.42f, paint);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(1.2f);
        paint.setColor(0x887B7460);
        canvas.drawCircle(cx, cy, Math.min(cx, cy) * 0.42f, paint);
    }
}
