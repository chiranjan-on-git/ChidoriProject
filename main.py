import cv2
import os
import time
import numpy as np
from src.hand_tracking import HandDetector
from src.utils import overlay_transparent, draw_aesthetic_text, draw_chidori_bar

# ─────────────────────────────────────────────────────────────────────────────
# Sound helper
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# Sound helper (Patched for Headerless & MP3 Pipeline)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pygame
    # Force 44.1kHz, 16-bit stereo PCM streaming with a stable buffer size
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    # Explicitly initialize the core framework so background channels sync
    pygame.init() 
    _PYGAME_OK = True
except Exception as e:
    print(f"[WARN] Failed to initialize Pygame Audio Core: {e}")
    _PYGAME_OK = False

def play_sound(path):
    if not _PYGAME_OK or not os.path.exists(path):
        print(f"[AUDIO] Audio file missing at target destination: {path}")
        return
    try:
        if path.endswith('.mp3'):
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        else:
            sound = pygame.mixer.Sound(path)
            sound.play()
    except Exception as e:
        print(f"[AUDIO ERROR] Pygame failed to route stream to device channels: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_SEQUENCE  = ["OX", "HARE", "MONKEY"]
SEQUENCE_TIMEOUT   = 5.0
SIGN_DEBOUNCE      = 0.8
CHIDORI_DURATION   = 8.0
CHIDORI_SOUND_PATH = "assets/sfx/chidori.mp3"
CHIDORI_IMG_PATH   = "assets/chidori.png"
EFFECT_SIZE        = (360, 360)  # Increased from 220 to 360
EFFECT_OFFSET      = (-180, -180) # Re-centered (-360/2, -360/2)

# Set True while testing — shows live orientation + finger readings on screen
DEBUG_MODE         = False


# ─────────────────────────────────────────────────────────────────────────────
# Load chidori image safely
# ─────────────────────────────────────────────────────────────────────────────
def _load_chidori(path):
    if not os.path.exists(path):
        print(f"[WARN] chidori image not found at: {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[WARN] cv2 could not read: {path}")
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

chidori_img = _load_chidori(CHIDORI_IMG_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Palm centroid
# ─────────────────────────────────────────────────────────────────────────────
def palm_centroid(landmarks):
    ids = [0, 5, 9, 13, 17]
    xs  = [landmarks[i][1] for i in ids]
    ys  = [landmarks[i][2] for i in ids]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


# ─────────────────────────────────────────────────────────────────────────────
# Debug overlay — shows raw readings so you can tune thresholds
# ─────────────────────────────────────────────────────────────────────────────
def render_debug(frame, detector, all_hands, detected_sign):
    import math
    h, w = frame.shape[:2]
    y = h - 130

    lines = [f"Detected: {detected_sign}", f"Hands: {len(all_hands)}"]

    for i, hand in enumerate(all_hands):
        lm = hand["landmarks"]

        dx = lm[9][1] - lm[0][1]
        dy = lm[9][2] - lm[0][2]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))

        ext = detector._count_extended_fingers(lm)
        spread = detector._fingers_spread(lm)
        scale  = detector._hand_scale(lm)
        spread_ratio = spread / scale if scale else 0

        ori = "VERT" if angle > 50 else ("HORIZ" if angle < 40 else "MID")
        lines.append(f"H{i}: {ori}({angle:.0f}deg) ext={ext} spread={spread_ratio:.2f}")

    if len(all_hands) == 2:
        import math
        lm0, lm1 = all_hands[0]["landmarks"], all_hands[1]["landmarks"]
        wd = math.hypot(lm0[0][1]-lm1[0][1], lm0[0][2]-lm1[0][2])
        avg_s = (detector._hand_scale(lm0) + detector._hand_scale(lm1)) / 2
        lines.append(f"WristDist={wd:.0f} AvgScale={avg_s:.0f} ratio={wd/avg_s:.2f}")

    for line in lines:
        cv2.rectangle(frame, (5, y - 18), (5 + len(line) * 10, y + 4), (0,0,0), -1)
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_PLAIN, 1.1,
                    (0, 255, 255), 1, cv2.LINE_AA)
        y += 22

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────────────────────────
SIGN_LABELS = {"OX": "#1  OX", "HARE": "#2  HARE", "MONKEY": "#3  MONKEY"}

def render_hud(frame, current_step, jutsu_activated, chidori_end_time):
    h, w = frame.shape[:2]

    if jutsu_activated:
        remaining = max(0.0, chidori_end_time - time.time())
        fraction  = remaining / CHIDORI_DURATION

        pulse    = int(60 + 40 * abs(np.sin(time.time() * 4)))
        vignette = np.zeros_like(frame, dtype=np.uint8)
        cv2.rectangle(vignette, (0, 0), (w, h), (pulse, 10, 0), -1)
        frame = cv2.addWeighted(frame, 1.0, vignette, 0.18, 0)

        draw_aesthetic_text(
            frame, "* C H I D O R I *",
            (w // 2 - 185, 62),
            font_scale=1.1, color=(255, 255, 255),
            glow_color=(0, 140, 255), thickness=2
        )
        draw_chidori_bar(frame, fraction,
                         bar_x=w // 2 - 150, bar_w=300, bar_y=h - 45)

    elif current_step > 0:
        done = "  >  ".join(SIGN_LABELS[s] for s in REQUIRED_SEQUENCE[:current_step])
        nxt  = SIGN_LABELS[REQUIRED_SEQUENCE[current_step]]
        draw_aesthetic_text(frame, done, (10, 45),
            font_scale=0.55, color=(200, 255, 200),
            glow_color=(0, 180, 80), thickness=1)
        draw_aesthetic_text(frame, f"Next >  {nxt}", (10, 85),
            font_scale=0.78, color=(255, 255, 120),
            glow_color=(0, 200, 200), thickness=2)
    else:
        draw_aesthetic_text(frame, "Form Sign  #1  OX  to begin", (10, 55),
            font_scale=0.75, color=(220, 220, 220),
            glow_color=(60, 60, 160), thickness=1)

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

    print("System active.  OX -> HARE -> MONKEY")
    if DEBUG_MODE:
        print("[DEBUG MODE ON] Live readings shown on screen. Set DEBUG_MODE=False when done tuning.")

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
        detected = "NONE"
        if all_hands and not jutsu_activated:
            detected = detector.verify_gesture(all_hands)

            if detected == REQUIRED_SEQUENCE[current_step]:
                if time.time() - last_sign_time > SIGN_DEBOUNCE or current_step == 0:
                    current_step  += 1
                    last_sign_time = time.time()
                    print(f"Sign confirmed: {detected}  ({current_step}/{len(REQUIRED_SEQUENCE)})")

                    if current_step == len(REQUIRED_SEQUENCE):
                        jutsu_activated  = True
                        chidori_end_time = time.time() + CHIDORI_DURATION
                        play_sound(CHIDORI_SOUND_PATH)
                        print("CHIDORI ACTIVATED!")

        # Chidori overlay
        # Chidori overlay (Dynamic Arm Scaling Mode)
        if jutsu_activated and chidori_img is not None and all_hands:
            lm     = all_hands[0]["landmarks"]
            cx, cy = palm_centroid(lm)
            
            # 1. Fetch the dynamic hand scale (Wrist to Middle Knuckle distance)
            current_hand_scale = detector._hand_scale(lm)
            
            # 2. Base size scales dynamically with your arm distance.
            # Multiplier 3.8 scales the graphic out to cover roughly 80% of your visible arm length.
            base_dynamic_size = current_hand_scale * 3.8
            
            # 3. Apply the lightning breathing/pulse effect to the dynamic base size
            pulse_scale = 1.0 + 0.06 * np.sin(time.time() * 8)
            dynamic_w = int(base_dynamic_size * pulse_scale)
            dynamic_h = int(base_dynamic_size * pulse_scale)
            
            # 4. Compute the dynamic offset automatically so the center is locked to your palm
            dynamic_offset_x = -int(dynamic_w / 2)
            dynamic_offset_y = -int(dynamic_h / 2)
            
            # 5. Calculate positioning and apply the transparency overlay
            ex, ey = cx + dynamic_offset_x, cy + dynamic_offset_y
            frame = overlay_transparent(frame, chidori_img, ex, ey, size=(dynamic_w, dynamic_h))

        frame = render_hud(frame, current_step, jutsu_activated, chidori_end_time)

        if DEBUG_MODE:
            frame = render_debug(frame, detector, all_hands, detected)

        cv2.imshow("Chidori Jutsu Training", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()