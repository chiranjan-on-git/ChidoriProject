import cv2
import os
import time
from src.hand_tracking import HandDetector

def main():
    cap = cv2.VideoCapture(0) # Adjust the index if you have multiple cameras
    detector = HandDetector(max_hands=2, detection_con=0.65)
    
    # Jutsu Sequence Tracking Configurations
    REQUIRED_SEQUENCE = ["OX", "HARE", "MONKEY"]
    current_step = 0
    jutsu_activated = False
    
    # Timing limits: You must hit the next sign within 3 seconds of the last one
    last_sign_time = time.time()
    SEQUENCE_TIMEOUT = 3.0 

    print("System active. Form the signs in order: OX -> HARE -> MONKEY")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = detector.find_hands(frame, draw=True)
        all_hands = detector.get_positions(frame)

        # Check for sequence timeouts to reset progress
        if current_step > 0 and not jutsu_activated:
            if time.time() - last_sign_time > SEQUENCE_TIMEOUT:
                print("Jutsu failed: Timeout! Resetting sequence.")
                current_step = 0

        if len(all_hands) > 0:
            detected_sign = detector.verify_gesture(all_hands)
            
            # If we are looking for the next sign in our list
            if not jutsu_activated and detected_sign == REQUIRED_SEQUENCE[current_step]:
                # Debounce: Make sure it's not matching old timestamps
                if time.time() - last_sign_time > 0.5 or current_step == 0:
                    current_step += 1
                    last_sign_time = time.time()
                    print(f"Sign confirmed: {detected_sign}! Move to step {current_step + 1}")
                    
                    # Check if the final step (MONKEY) was achieved
                    if current_step == len(REQUIRED_SEQUENCE):
                        jutsu_activated = True
                        print("⚡ CHIDORI ACTIVATED! ⚡")

            # Allow an intentional manual reset by forming a wrong shape or hiding hands
            if jutsu_activated and detected_sign == "NONE" and (time.time() - last_sign_time > 5.0):
                jutsu_activated = False
                current_step = 0
                print("Chidori dissipated. Ready for next attempt.")

        # HUD Text Overlay Rendering
        if jutsu_activated:
            status_text = "JUTSU: CHIDORI ACTIVATED!"
            text_color = (255, 0, 0) # Striking Blue profile
        elif current_step > 0:
            status_text = f"Sequence: {' -> '.join(REQUIRED_SEQUENCE[:current_step])}"
            text_color = (0, 255, 255) # Yellow profile
        else:
            status_text = "Form sign: OX to begin"
            text_color = (0, 0, 255) # Red profile

        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        cv2.imshow("Chidori Jutsu Training", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()