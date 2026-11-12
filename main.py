
import cv2
import time
import os
import sys
from ultralytics import YOLO


def resource_path(filename):
    """Resolve a bundled resource in both source and PyInstaller builds."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, filename)

# --------------------------------
# Configuration
# --------------------------------

cap = cv2.VideoCapture(0)

# Mirror webcam so movement matches what you see
MIRROR_CAMERA = True

model = YOLO(resource_path("yolo26n.pt"))

in_count = 0
out_count = 0

LINE_X = None

DEAD_ZONE = 25
CONFIRM_FRAMES = 3
SMOOTHING = 0.35
MAX_LOST_FRAMES = 90

tracks = {}
frame_number = 0


# --------------------------------
# Initialize tracking state
# --------------------------------

def new_track(x):

    return {
        "smooth_x": x,
        "stable_side": 0,
        "candidate_side": 0,
        "candidate_frames": 0,
        "last_seen": frame_number
    }


def get_side(x):

    if x < LINE_X - DEAD_ZONE:
        return -1  # LEFT

    if x > LINE_X + DEAD_ZONE:
        return 1   # RIGHT

    return 0      # DEAD ZONE


# --------------------------------
# Main loop
# --------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    if MIRROR_CAMERA:
        frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    LINE_X = width // 2

    # --------------------------------
    # YOLO + ByteTrack
    # --------------------------------

    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.30,
        imgsz=960,
        verbose=False
    )[0]

    output = result.plot()

    # --------------------------------
    # Process tracked people
    # --------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Bounding box center
            center_x = (x1 + x2) // 2

            # Create state for new person
            if track_id not in tracks:
                tracks[track_id] = new_track(center_x)

            state = tracks[track_id]

            state["last_seen"] = frame_number

            # --------------------------------
            # Smooth position
            # --------------------------------

            state["smooth_x"] = (
                SMOOTHING * center_x
                + (1 - SMOOTHING) * state["smooth_x"]
            )

            x = state["smooth_x"]

            current_side = get_side(x)

            # Ignore dead zone
            if current_side == 0:
                continue

            # --------------------------------
            # Establish initial position
            # --------------------------------

            if state["stable_side"] == 0:

                if current_side != state["candidate_side"]:
                    state["candidate_side"] = current_side
                    state["candidate_frames"] = 1
                else:
                    state["candidate_frames"] += 1

                if state["candidate_frames"] >= CONFIRM_FRAMES:
                    state["stable_side"] = current_side
                    state["candidate_frames"] = 0

                continue

            # --------------------------------
            # Check direction
            # --------------------------------

            if current_side == state["stable_side"]:

                state["candidate_side"] = 0
                state["candidate_frames"] = 0
                continue

            if current_side != state["candidate_side"]:

                state["candidate_side"] = current_side
                state["candidate_frames"] = 1

            else:

                state["candidate_frames"] += 1

            # Confirm crossing
            if state["candidate_frames"] >= CONFIRM_FRAMES:

                old_side = state["stable_side"]
                new_side = current_side

                # LEFT -> RIGHT = IN
                if old_side == -1 and new_side == 1:
                    in_count += 1
                    print(f"IN: {track_id}")

                # RIGHT -> LEFT = OUT
                elif old_side == 1 and new_side == -1:
                    out_count += 1
                    print(f"OUT: {track_id}")

                # Update state to allow repeated crossings
                state["stable_side"] = new_side
                state["candidate_side"] = 0
                state["candidate_frames"] = 0

    # --------------------------------
    # Cleanup stale tracks
    # --------------------------------

    expired = [
        track_id
        for track_id, state in tracks.items()
        if frame_number - state["last_seen"] > MAX_LOST_FRAMES
    ]

    for track_id in expired:
        del tracks[track_id]

    # --------------------------------
    # Display reference line
    # --------------------------------

    cv2.line(
        output,
        (LINE_X, 0),
        (LINE_X, height),
        (255, 255, 255),
        1
    )

    # --------------------------------
    # Counter display - black font
    # --------------------------------

    cv2.rectangle(
        output,
        (10, 10),
        (250, 115),
        (255, 255, 255),
        -1
    )

    cv2.putText(
        output,
        f"IN: {in_count}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )

    cv2.putText(
        output,
        f"OUT: {out_count}",
        (25, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )

    cv2.imshow("People Counter", output)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()

print("Total IN:", in_count)
print("Total OUT:", out_count)
