import cv2
import os
import time
import numpy as np
from src.hand_tracking import HandDetector
from src.utils import overlay_transparent, draw_aesthetic_text, draw_chidori_bar

# ─────────────────────────────────────────────────────────────────────────────
# Sound helper  (uses pygame if available, otherwise silent)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    _PYGAME_OK = True
except Exception:
    _PYGAME_OK = False

def play_sound(path):
    if not _PYGAME_OK or not os.path.exists(path):
        return
    try:
        pygame.mixer.Sound(path).play()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  ← tweak these freely
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_SEQUENCE  = ["OX", "HARE", "MONKEY"]
SEQUENCE_TIMEOUT   = 4.0   # seconds — reset if next sign isn't shown in time
SIGN_DEBOUNCE      = 0.6   # seconds — min gap between two accepted signs
CHIDORI_DURATION   = 8.0   # seconds — how long Chidori stays active
CHIDORI_SOUND_PATH = "assets/sfx/chidori.wav"
CHIDORI_IMG_PATH   = "assets/chidori.png"

EFFECT_SIZE        = (220, 220)    # (width, height) of the lightning overlay
EFFECT_OFFSET      = (-110, -130)  # pixel shift from palm centroid (x, y)


# ─────────────────────────────────────────────────────────────────────────────
# Load assets
# ─────────────────────────────────────────────────────────────────────────────
chidori_img = None
if os.path.exists(CHIDORI_IMG_PATH):
    chidori_img = cv2.imread(CHIDORI_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if chidori_img is not None and chidori_img.shape[2] < 4:
        chidori_img = cv2.cvtColor(chidori_img, cv2.COLOR_BGR2BGRA)


# ─────────────────────────────────────────────────────────────────────────────
# Palm centroid helper
# ─────────────────────────────────────────────────────────────────────────────
def palm_centroid(landmarks):
    """Average position of wrist + 4 knuckle bases."""
    ids = [0, 5, 9, 13, 17]
    xs  = [landmarks[i][1] for i in ids]
    ys  = [landmarks[i][2] for i in ids]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
SIGN_LABELS = {
    "OX":     "#1  OX",
    "HARE":   "#2  HARE",
    "MONKEY": "#3  MONKEY",
}

def render_hud(frame, current_step, jutsu_activated, chidori_end_time):
    h, w = frame.shape[:2]

    if jutsu_activated:
        remaining = max(0.0, chidori_end_time - time.time())
        fraction  = remaining / CHIDORI_DURATION

        # Pulsing blue vignette
        pulse    = int(60 + 40 * abs(np.sin(time.time() * 4)))
        vignette = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(vignette, (0, 0), (w, h), (pulse, 10, 0), -1)
        frame = cv2.addWeighted(frame, 1.0, vignette, 0.18, 0)

        draw_aesthetic_text(
            frame, "* C H I D O R I *",
            (w // 2 - 185, 62),
            font_scale=1.1,
            color=(255, 255, 255),
            glow_color=(0, 140, 255),
            thickness=2
        )
        draw_chidori_bar(frame, fraction,
                         bar_x=w // 2 - 150, bar_w=300, bar_y=h - 45)

    elif current_step > 0:
        done = "  >  ".join(SIGN_LABELS[s] for s in REQUIRED_SEQUENCE[:current_step])
        nxt  = SIGN_LABELS[REQUIRED_SEQUENCE[current_step]]

        draw_aesthetic_text(
            frame, done, (10, 45),
            font_scale=0.55,
            color=(200, 255, 200),
            glow_color=(0, 180, 80),
            thickness=1
        )
        draw_aesthetic_text(
            frame, f"Next >  {nxt}", (10, 85),
            font_scale=0.78,
            color=(255, 255, 120),
            glow_color=(0, 200, 200),
            thickness=2
        )

    else:
        draw_aesthetic_text(
            frame, "Form Sign  #1  OX  to begin", (10, 55),
            font_scale=0.75,
            color=(220, 220, 220),
            glow_color=(60, 60, 160),
            thickness=1
        )

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cap      = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=2, detection_con=0.65)

    current_step     = 0
    jutsu_activated  = False
    chidori_end_time = 0.0
    last_sign_time   = time.time()

    print("System active.  OX  →  HARE  →  MONKEY")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        frame     = detector.find_hands(frame, draw=True)
        all_hands = detector.get_positions(frame)

        # Auto-expire Chidori
        if jutsu_activated and time.time() > chidori_end_time:
            jutsu_activated = False
            current_step    = 0
            print("Chidori dissipated. Ready.")

        # Sequence timeout
        if current_step > 0 and not jutsu_activated:
            if time.time() - last_sign_time > SEQUENCE_TIMEOUT:
                print("Timeout — resetting.")
                current_step = 0

        # Gesture check
        if all_hands and not jutsu_activated:
            detected = detector.verify_gesture(all_hands)

            if detected == REQUIRED_SEQUENCE[current_step]:
                if time.time() - last_sign_time > SIGN_DEBOUNCE or current_step == 0:
                    current_step  += 1
                    last_sign_time = time.time()
                    print(f"✔ {detected}  ({current_step}/{len(REQUIRED_SEQUENCE)})")

                    if current_step == len(REQUIRED_SEQUENCE):
                        jutsu_activated  = True
                        chidori_end_time = time.time() + CHIDORI_DURATION
                        play_sound(CHIDORI_SOUND_PATH)
                        print("⚡ CHIDORI ACTIVATED! ⚡")

        # Chidori lightning overlay on palm
        if jutsu_activated and chidori_img is not None and all_hands:
            lm         = all_hands[0]["landmarks"]
            cx, cy     = palm_centroid(lm)
            ex, ey     = cx + EFFECT_OFFSET[0], cy + EFFECT_OFFSET[1]
            pulse_scale = 1.0 + 0.06 * np.sin(time.time() * 8)
            pw, ph      = int(EFFECT_SIZE[0] * pulse_scale), int(EFFECT_SIZE[1] * pulse_scale)
            frame       = overlay_transparent(frame, chidori_img, ex, ey, size=(pw, ph))

        # HUD
        frame = render_hud(frame, current_step, jutsu_activated, chidori_end_time)

        cv2.imshow("Chidori Jutsu Training", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()