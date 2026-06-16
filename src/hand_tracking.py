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

    # ─────────────────────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────────────────────
    def find_hands(self, frame, draw=True):
        img_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.latest_result = self.detector.detect(mp_image)

        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (0,9),(9,10),(10,11),(11,12),
            (0,13),(13,14),(14,15),(15,16),
            (0,17),(17,18),(18,19),(19,20),
            (5,9),(9,13),(13,17),(0,17)
        ]

        if draw and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for s, e in connections:
                    cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    # Position extraction
    # ─────────────────────────────────────────────────────────────────────────
    def get_positions(self, frame):
        all_hands = []
        if not (self.latest_result and self.latest_result.hand_landmarks):
            return all_hands

        h, w, _ = frame.shape
        for i, hand_landmarks in enumerate(self.latest_result.hand_landmarks):
            lm_list = []
            for idx, lm in enumerate(hand_landmarks):
                lm_list.append([idx, int(lm.x * w), int(lm.y * h), lm.z])

            handedness = "Unknown"
            if self.latest_result.handedness and i < len(self.latest_result.handedness):
                handedness = self.latest_result.handedness[i][0].display_name
            all_hands.append({"landmarks": lm_list, "label": handedness})
        return all_hands

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _dist(a, b):
        return math.hypot(a[1] - b[1], a[2] - b[2])

    @staticmethod
    def _hand_scale(lm):
        """Wrist(0) → middle-MCP(9) as reference length."""
        return math.hypot(lm[0][1] - lm[9][1], lm[0][2] - lm[9][2]) or 1.0

    @staticmethod
    def _orientation_angle(lm):
        """
        Angle (degrees) of wrist→middle-MCP vector.
        0° = perfectly horizontal, 90° = perfectly vertical.
        """
        dx = lm[9][1] - lm[0][1]
        dy = lm[9][2] - lm[0][2]
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    @staticmethod
    def _is_vertical(lm, threshold=50):
        """True when wrist→MCP angle > threshold (hand pointing up/down)."""
        dx = lm[9][1] - lm[0][1]
        dy = lm[9][2] - lm[0][2]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        return angle > threshold

    @staticmethod
    def _is_horizontal(lm, threshold=40):
        """True when wrist→MCP angle < threshold (hand pointing sideways)."""
        dx = lm[9][1] - lm[0][1]
        dy = lm[9][2] - lm[0][2]
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        return angle < threshold

    @staticmethod
    def _count_extended_fingers(lm):
        """
        Count how many of index/middle/ring/pinky are extended.
        Extended = tip is higher on screen (lower y) than the PIP joint.
        """
        count = 0
        for tip_id in [8, 12, 16, 20]:
            if lm[tip_id][2] < lm[tip_id - 2][2]:
                count += 1
        return count

    @staticmethod
    def _fingers_spread(lm):
        """
        Measures the angular spread of the four fingertips around the palm.
        High spread = fingers fanned open wide.
        Returns the average pixel distance between adjacent fingertips.
        """
        tips = [lm[8], lm[12], lm[16], lm[20]]
        gaps = []
        for i in range(len(tips) - 1):
            gaps.append(math.hypot(tips[i][1] - tips[i+1][1],
                                   tips[i][2] - tips[i+1][2]))
        return sum(gaps) / len(gaps) if gaps else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Gesture classifier (Now nested correctly inside the class)
    # ─────────────────────────────────────────────────────────────────────────
    def verify_gesture(self, all_hands):
        if not all_hands or len(all_hands) < 2:
            return "NONE"

        try:
            lm0 = all_hands[0]["landmarks"]
            lm1 = all_hands[1]["landmarks"]

            # Prevent IndexErrors if landmarks drop out unexpectedly
            if len(lm0) < 21 or len(lm1) < 21:
                return "NONE"

            scale0 = self._hand_scale(lm0)
            scale1 = self._hand_scale(lm1)
            avg_scale = (scale0 + scale1) / 2.0

            wrist_dist = self._dist(lm0[0], lm1[0])
            ratio = wrist_dist / avg_scale

            horiz0 = self._is_horizontal(lm0)
            horiz1 = self._is_horizontal(lm1)
            vert0 = self._is_vertical(lm0)
            vert1 = self._is_vertical(lm1)

            spread0 = self._fingers_spread(lm0)
            spread1 = self._fingers_spread(lm1)
            
            spread_ratio0 = spread0 / scale0 if scale0 else 0
            spread_ratio1 = spread1 / scale1 if scale1 else 0

            # ── 1. MONKEY/HARE Stance: Both flat horizontal ──
            if horiz0 and horiz1:
                # If hands are stacked flat but wrist distance ratio is closer
                if ratio < 1.8:
                    return "HARE"
                
                # Both horizontal, but arms/wrists wider apart with open fingers
                elif ratio >= 1.8:
                    if spread_ratio0 > 0.18 or spread_ratio1 > 0.18:
                        return "MONKEY"

            # ── 2. Cross Shape Fallbacks (OX) ──
            is_cross = (vert0 and horiz1) or (vert1 and horiz0)
            if is_cross and ratio < 2.5:
                if vert0:
                    v_spread, v_scale = spread0, scale0
                else:
                    v_spread, v_scale = spread1, scale1
                    
                s_ratio = v_spread / v_scale if v_scale else 0
                if s_ratio > 0.35:
                    return "MONKEY"
                else:
                    return "OX"

        except Exception as e:
            # Fallback that prints the issue without breaking the CV2 webcam loop
            print(f"[SAFEMODE] Catching background anomaly: {e}")
            return "NONE"

        return "NONE"