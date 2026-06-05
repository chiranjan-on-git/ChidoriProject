import cv2
import numpy as np

def overlay_transparent(background, overlay, x, y, size=None):
    """
    Overlays a transparent PNG onto a background image at position (x, y).
    Optionally resizes the overlay image to 'size' (width, height).
    """
    if size is not None:
        overlay = cv2.resize(overlay, size)

    h, w, c = overlay.shape
    if c < 4:
        # If the image doesn't have an alpha channel, we can't do transparency easily
        return background

    # Isolate the background region of interest (ROI)
    bg_h, bg_w, _ = background.shape
    if y < 0 or x < 0 or y + h > bg_h or x + w > bg_w:
        return background  # Prevent crashing if the hand moves off-screen

    roi = background[y:y+h, x:x+w]

    # Split channels
    overlay_img = overlay[:, :, :3]
    mask = overlay[:, :, 3] / 255.0
    mask_inv = 1.0 - mask

    # Blend overlay and ROI channel by channel
    for c_idx in range(3):
        roi[:, :, c_idx] = (mask * overlay_img[:, :, c_idx] + mask_inv * roi[:, :, c_idx])

    background[y:y+h, x:x+w] = roi
    return background