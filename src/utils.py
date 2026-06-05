import cv2
import numpy as np


def overlay_transparent(background, overlay, x, y, size=None):
    """
    Overlays a transparent PNG onto a background image at position (x, y).
    Optionally resizes the overlay image to 'size' (width, height).
    """
    if size is not None:
        overlay = cv2.resize(overlay, size)

    h, w = overlay.shape[:2]
    if overlay.ndim < 3 or overlay.shape[2] < 4:
        # No alpha channel – plain blit
        bg_h, bg_w = background.shape[:2]
        y2, x2 = min(y + h, bg_h), min(x + w, bg_w)
        background[max(y, 0):y2, max(x, 0):x2] = overlay[:y2-y, :x2-x]
        return background

    bg_h, bg_w = background.shape[:2]
    # Clamp to screen bounds
    if y >= bg_h or x >= bg_w or y + h <= 0 or x + w <= 0:
        return background

    # Crop overlay and target rect if partially off-screen
    oy1 = max(0, -y);    oy2 = h - max(0, (y + h) - bg_h)
    ox1 = max(0, -x);    ox2 = w - max(0, (x + w) - bg_w)
    ty1 = max(0, y);     ty2 = ty1 + (oy2 - oy1)
    tx1 = max(0, x);     tx2 = tx1 + (ox2 - ox1)

    if ty2 <= ty1 or tx2 <= tx1:
        return background

    roi           = background[ty1:ty2, tx1:tx2].astype(np.float32)
    overlay_crop  = overlay[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    alpha         = overlay[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0

    blended = overlay_crop * alpha + roi * (1.0 - alpha)
    background[ty1:ty2, tx1:tx2] = blended.astype(np.uint8)
    return background


def draw_aesthetic_text(frame, text, pos, font_scale=0.9, thickness=2,
                        color=(255, 255, 255), glow_color=(0, 200, 255)):
    """
    Draws text with a neon-glow effect by layering blurred and sharp renders.
    """
    x, y = pos
    font = cv2.FONT_HERSHEY_DUPLEX

    # Glow layers (larger, blurred behind)
    for spread in [6, 4, 2]:
        cv2.putText(frame, text, (x, y), font, font_scale,
                    glow_color, thickness + spread, cv2.LINE_AA)

    # Core sharp text
    cv2.putText(frame, text, (x, y), font, font_scale,
                color, thickness, cv2.LINE_AA)

    return frame


def draw_chidori_bar(frame, fraction, bar_x=20, bar_y=None,
                     bar_w=300, bar_h=14):
    """
    Draws a glowing blue progress bar representing remaining Chidori time.
    fraction: 0.0 (empty) → 1.0 (full)
    """
    h, w = frame.shape[:2]
    if bar_y is None:
        bar_y = h - 40

    # Background track
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (20, 20, 40), -1)

    # Filled portion
    fill_w = int(bar_w * max(0.0, min(1.0, fraction)))
    if fill_w > 0:
        # Outer glow
        for expand in [4, 2]:
            cv2.rectangle(frame,
                          (bar_x - expand, bar_y - expand),
                          (bar_x + fill_w + expand, bar_y + bar_h + expand),
                          (0, 120, 255), -1)
        # Inner bright fill
        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + fill_w, bar_y + bar_h),
                      (180, 240, 255), -1)

    # Border
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (0, 160, 255), 1)

    return frame