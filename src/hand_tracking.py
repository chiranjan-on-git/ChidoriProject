import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandDetector:
    def __init__(self, model_path="assets/hand_landmarker.task", max_hands=2, detection_con=0.65):
        # Configure the modern MediaPipe Task options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_con,
            running_mode=vision.RunningMode.IMAGE
        )
        # Create the instance handler
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.tip_ids = [8, 12, 16, 20]
        self.latest_result = None

    def find_hands(self, frame, draw=True):
        # Format BGR Frame to RGB representation for processing
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Execute the modern detection API
        self.latest_result = self.detector.detect(mp_image)

        # Hardcoded rendering connections layout mapping
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),# Ring
            (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (5, 9), (9, 13), (13, 17), (0, 17)   # Palm boundaries
        ]

        if draw and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                # Map coordinates back into absolute pixel vectors
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                
                # Draw the skeletal links
                for start, end in connections:
                    cv2.line(frame, pts[start], pts[end], (0, 255, 0), 2)
                # Draw joint coordinates points
                for pt in pts:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                    
        return frame

    def get_positions(self, frame):
        """Extracts structured coordinates maps using the new object schemas."""
        all_hands = []
        if self.latest_result and self.latest_result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in self.latest_result.hand_landmarks:
                landmark_list = []
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmark_list.append([idx, cx, cy])
                all_hands.append(landmark_list)
        return all_hands

    def verify_gesture(self, all_hands):
        """Analyzes active coordinates to detect Jutsu shapes."""
        if len(all_hands) == 0:
            return "NONE"

        hand1 = all_hands[0]
        fingers = []

        # Thumb calculation checking
        if hand1[4][1] > hand1[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Standard fingers extension check
        for id in self.tip_ids:
            if hand1[id][2] < hand1[id - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # 1. MONKEY SIGN
        if fingers[1:] == [1, 1, 1, 1] and abs(hand1[8][2] - hand1[12][2]) < 30:
            return "MONKEY"

        # 2. HARE SIGN
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            return "HARE"

        # 3. OX SIGN
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
            return "OX"

        return "NONE"