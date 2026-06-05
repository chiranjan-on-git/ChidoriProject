import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math


class HandDetector:
    def __init__(self, model_path="assets/hand_landmarker.task", max_hands=2, detection_con=0.65):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.latest_result = None

    # ──────────────────────────────────────────────────────────────────────────
    # Drawing
    # ──────────────────────────────────────────────────────────────────────────
    def find_hands(self, frame, draw=True):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.latest_result = self.detector.detect(mp_image)

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17), (0, 17)
        ]

        if draw and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for start, end in connections:
                    cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

        return frame

    # ──────────────────────────────────────────────────────────────────────────
    # Position extraction
    # ──────────────────────────────────────────────────────────────────────────
    def get_positions(self, frame):
        all_hands = []
        if self.latest_result and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for i, hand_landmarks in enumerate(self.latest_result.hand_landmarks):
                landmark_list = []
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                    landmark_list.append([idx, cx, cy, cz])
                handedness = "Unknown"
                if self.latest_result.handedness and i < len(self.latest_result.handedness):
                    handedness = self.latest_result.handedness[i][0].display_name
                all_hands.append({"landmarks": landmark_list, "label": handedness})
        return all_hands

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _dist(self, p1, p2):
        """2-D pixel distance between two landmark entries [id, cx, cy, cz]."""
        return math.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def _get_fingers(self, lm):
        """
        Returns [thumb, index, middle, ring, pinky] — 1=extended, 0=folded.
        Works from any camera angle by using the wrist→index-MCP vector for thumb.
        """
        fingers = []

        wrist     = lm[0]
        idx_mcp   = lm[5]
        thumb_tip = lm[4]
        thumb_ip  = lm[3]

        hand_dx  = idx_mcp[1] - wrist[1]
        thumb_dx = thumb_tip[1] - thumb_ip[1]
        fingers.append(1 if hand_dx * thumb_dx > 0 else 0)

        for tip_id in [8, 12, 16, 20]:
            fingers.append(1 if lm[tip_id][2] < lm[tip_id - 2][2] else 0)

        return fingers  # [thumb, index, middle, ring, pinky]

    # ──────────────────────────────────────────────────────────────────────────
    # Sign detection — 3-sign Chidori (OX → HARE → MONKEY)
    # ──────────────────────────────────────────────────────────────────────────
    def verify_gesture(self, all_hands):
        """
        OX    (1 hand)  – index + middle both extended and crossed close together;
                          ring + pinky folded; thumb out to the side.
        HARE  (2 hands) – both hands visible; both index fingers extended (pointing
                          forward); remaining fingers mostly curled on each hand;
                          the two hands are close / overlapping (clasped pose).
        MONKEY(2 hands) – both hands visible; all fingers on both hands extended
                          (open flat palms) and the two wrists are close together
                          (hands held side-by-side / interlocked horizontally).
        """
        if not all_hands:
            return "NONE"

        num_hands = len(all_hands)

        # ── Single-hand signs ────────────────────────────────────────────────
        lm1     = all_hands[0]["landmarks"]
        f1      = self._get_fingers(lm1)
        _, idx1, mid1, ring1, pinky1 = f1

        hand_scale1 = self._dist(lm1[0], lm1[9])   # wrist → mid-MCP

        # OX: index + middle up, crossed close together; ring + pinky down
        idx_mid_gap = self._dist(lm1[8], lm1[12])   # index tip vs middle tip
        if (idx1 == 1 and mid1 == 1 and ring1 == 0 and pinky1 == 0
                and idx_mid_gap < hand_scale1 * 0.35):
            return "OX"

        # ── Two-hand signs ───────────────────────────────────────────────────
        if num_hands < 2:
            return "NONE"

        lm2     = all_hands[1]["landmarks"]
        f2      = self._get_fingers(lm2)
        _, idx2, mid2, ring2, pinky2 = f2

        hand_scale2   = self._dist(lm2[0], lm2[9])
        # Distance between the two wrists
        wrist_gap     = self._dist(lm1[0], lm2[0])
        avg_scale     = (hand_scale1 + hand_scale2) / 2.0

        # MONKEY: all fingers open on BOTH hands + wrists close together (side by side)
        both_open = (idx1 == 1 and mid1 == 1 and ring1 == 1 and pinky1 == 1 and
                     idx2 == 1 and mid2 == 1 and ring2 == 1 and pinky2 == 1)
        if both_open and wrist_gap < avg_scale * 2.2:
            return "MONKEY"

        # HARE: index fingers up on BOTH hands, other fingers curled,
        #        and hands held close / overlapping (clasped)
        both_pointing = (idx1 == 1 and mid1 == 0 and ring1 == 0 and pinky1 == 0 and
                         idx2 == 1 and mid2 == 0 and ring2 == 0 and pinky2 == 0)
        if both_pointing and wrist_gap < avg_scale * 2.5:
            return "HARE"

        return "NONE"