# Chidori Project

A real-time computer vision project that lets you activate the **Chidori** effect from *Naruto* by performing a sequence of hand signs in front of your webcam.

The application uses OpenCV and MediaPipe to track both hands, recognizes the required gesture sequence, and then overlays a Chidori lightning effect on the camera feed with a sound effect and an on-screen timer.

## Features

- Tracks up to two hands using MediaPipe Hand Landmarker.
- Recognizes the hand-sign sequence: **OX → HARE → MONKEY**.
- Displays hand landmarks and a simple progress HUD.
- Activates a Chidori image overlay when the full sequence is completed.
- Scales and positions the effect relative to the detected hand.
- Plays a Chidori sound effect and displays an 8-second duration bar.
- Includes an optional debug overlay for gesture-tuning.

## Project Structure

```text
ChidoriProject/
├── assets/
│   ├── chidori.png
│   ├── hand_landmarker.task
│   └── sfx/
│       └── chidori.mp3
├── src/
│   ├── __init__.py
│   ├── hand_tracking.py
│   └── utils.py
├── main.py
└── requirements.txt
```

Additional image and audio assets may also be present in the project.

## Requirements

- Python 3.9 or later (use a Python version supported by your installed MediaPipe release).
- A working webcam.
- A microphone is **not** required.
- The following Python packages:
  - OpenCV
  - MediaPipe
  - NumPy
  - Pygame (for audio playback)

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows:**
   ```bash
   .venv\Scripts\activate
   ```

   **macOS / Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Install Pygame for the Chidori sound effect:

   ```bash
   pip install pygame
   ```

## Run the Project

From the project root directory, run:

```bash
python main.py
```

Allow the application to access your webcam if prompted. Hold both hands in view and perform the signs in this order:

1. **OX**
2. **HARE**
3. **MONKEY**

When the complete sequence is recognized, the Chidori effect and sound will activate. The effect runs for approximately 8 seconds, after which the application resets and is ready for another sequence.

Press **Q** while the camera window is focused to quit.

## Configuration

The main settings are defined near the top of `main.py`:

| Setting | Default | Description |
|---|---:|---|
| `REQUIRED_SEQUENCE` | `["OX", "HARE", "MONKEY"]` | Required order of hand signs |
| `SEQUENCE_TIMEOUT` | `5.0` | Seconds allowed between signs before the sequence resets |
| `SIGN_DEBOUNCE` | `0.8` | Minimum interval between accepted signs |
| `CHIDORI_DURATION` | `8.0` | Duration of the activated effect, in seconds |
| `DEBUG_MODE` | `False` | Enables live gesture measurements on screen |

The hand detector is configured in `src/hand_tracking.py`. It uses two hands and a default detection confidence of `0.65`.

## Debugging

If a hand sign is not being recognized:

1. Set `DEBUG_MODE = True` in `main.py`.
2. Run the project again.
3. Keep both hands fully visible and well-lit.
4. Watch the on-screen orientation, finger-spread, and wrist-distance readings.
5. Adjust the thresholds in `src/hand_tracking.py` if needed.

Set `DEBUG_MODE = False` when you finish tuning.

## Troubleshooting

- **Webcam does not open:** Check that the webcam is connected and that no other application is using it. The project currently opens camera device `0` in `main.py`.
- **Hand landmarks are not detected:** Improve lighting, move your hands into the camera frame, and check that `assets/hand_landmarker.task` exists.
- **Chidori image is missing:** Check that `assets/chidori.png` exists. The application can continue without the visual effect image, but the overlay will not appear.
- **No sound:** Confirm that Pygame is installed, your audio output is available, and `assets/sfx/chidori.mp3` exists. Audio initialization failures are reported as warnings.
- **Missing Python modules:** Activate your virtual environment and run `pip install -r requirements.txt`, followed by `pip install pygame`.

Run commands from the project root so the relative paths to the assets resolve correctly.

## How It Works

1. OpenCV captures frames from the webcam and flips them horizontally.
2. MediaPipe detects hand landmarks for up to two hands.
3. The hand detector estimates hand orientation, finger spread, and the distance between wrists to classify the supported signs.
4. The main loop checks the signs against the required sequence, applying a timeout and debounce interval.
5. After all three signs are recognized, the application plays the sound and overlays the Chidori graphic near the detected palm.
6. The HUD displays the activation state and remaining duration before resetting.

## Notes

- Gesture recognition is heuristic and may depend on camera angle, lighting, hand position, and individual movement.
- Keep both hands visible for reliable recognition.
- Asset paths are relative to the project root.
- The project is a fan-made computer vision demonstration inspired by *Naruto*.

## License

No license file was included with the supplied project. Add a `LICENSE` file if you intend to distribute the project and want to specify reuse terms. Ensure you have the appropriate rights to any third-party images, sounds, and model assets you distribute.
