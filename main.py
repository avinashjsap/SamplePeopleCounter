
import cv2
import csv
import logging
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("people_counter")


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
GENDER_CLASSIFY_INTERVAL = 12
GENDER_CONFIDENCE_THRESHOLD = 0.55
AGE_CLASSIFY_INTERVAL = 12
AGE_FACE_CONFIDENCE_THRESHOLD = 0.5
AGE_MIN_FACE_SIZE = 20
TRACKING_LOG_PATH = Path.home() / "Documents" / "PeopleCounter" / "tracking_log.csv"
TRACKING_SESSION_ID = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
CSV_LOGGING_ENABLED = False
TRACKING_LOG_FIELDS = (
    "timestamp",
    "session_id",
    "track_id",
    "gender",
    "gender_confidence",
    "age_estimate",
    "age_confidence",
)
if CSV_LOGGING_ENABLED:
    logger.info("Tracking CSV: %s", TRACKING_LOG_PATH)
else:
    logger.info("CSV logging is disabled; console logging is active.")

GENDER_MODEL_PATH = resource_path("gender_classification.pt")
gender_model = YOLO(GENDER_MODEL_PATH) if os.path.isfile(GENDER_MODEL_PATH) else None
if gender_model is None:
    logger.warning(
        "Gender model not found. Add gender_classification.pt "
        "beside the app to enable person labels."
    )

AGE_BUCKETS = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
FACE_DETECTOR_CONFIG = resource_path("face_detector/deploy.prototxt")
FACE_DETECTOR_WEIGHTS = resource_path(
    "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
AGE_MODEL_CONFIG = resource_path("age_detector/age_deploy.prototxt")
AGE_MODEL_WEIGHTS = resource_path("age_detector/age_net.caffemodel")

face_net = None
age_net = None
age_model_files = (
    FACE_DETECTOR_CONFIG,
    FACE_DETECTOR_WEIGHTS,
    AGE_MODEL_CONFIG,
    AGE_MODEL_WEIGHTS,
)
if all(os.path.isfile(path) for path in age_model_files):
    face_net = cv2.dnn.readNetFromCaffe(FACE_DETECTOR_CONFIG, FACE_DETECTOR_WEIGHTS)
    age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL_CONFIG, AGE_MODEL_WEIGHTS)
else:
    logger.warning(
        "Age model files missing. Add the files under face_detector/ and age_detector/."
    )

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
        "last_seen": frame_number,
        "gender_label": None,
        "gender_confidence": 0.0,
        "gender_last_frame": frame_number - GENDER_CLASSIFY_INTERVAL,
        "age_range": None,
        "age_confidence": 0.0,
        "age_last_frame": frame_number - AGE_CLASSIFY_INTERVAL,
        "last_logged_signature": None,
    }


def log_track_update(track_id, state):
    """Append a row when a track's predicted gender or age label changes."""
    signature = (state["gender_label"], state["age_range"])
    if signature == state["last_logged_signature"] or all(value is None for value in signature):
        return

    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "session_id": TRACKING_SESSION_ID,
        "track_id": track_id,
        "gender": state["gender_label"] or "",
        "gender_confidence": (
            f"{state['gender_confidence']:.4f}" if state["gender_label"] else ""
        ),
        "age_estimate": state["age_range"] or "",
        "age_confidence": (
            f"{state['age_confidence']:.4f}" if state["age_range"] else ""
        ),
    }
    state["last_logged_signature"] = signature
    logger.info(
        "track update id=%s gender=%s gender_confidence=%s age=%s age_confidence=%s",
        track_id,
        row["gender"] or "n/a",
        row["gender_confidence"] or "n/a",
        row["age_estimate"] or "n/a",
        row["age_confidence"] or "n/a",
    )

    if CSV_LOGGING_ENABLED:
        try:
            TRACKING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_exists = TRACKING_LOG_PATH.exists()
            with TRACKING_LOG_PATH.open("a", newline="", encoding="utf-8") as log_file:
                writer = csv.DictWriter(log_file, fieldnames=TRACKING_LOG_FIELDS)
                if not file_exists or TRACKING_LOG_PATH.stat().st_size == 0:
                    writer.writeheader()
                writer.writerow(row)
        except OSError:
            logger.exception("Could not write tracking log %s", TRACKING_LOG_PATH)


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

    # Draw boxes first; labels are rebuilt below to combine person and gender
    # text into a single Ultralytics-style label.
    has_track_ids = result.boxes is not None and result.boxes.id is not None
    output = (
        result.plot(conf=False, labels=False)
        if has_track_ids
        else result.plot()
    )
    annotator = Annotator(output)

    # --------------------------------
    # Process tracked people
    # --------------------------------

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()
        box_confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, person_class_id, box_confidence in zip(
            boxes, ids, class_ids, box_confidences
        ):

            x1, y1, x2, y2 = map(int, box)

            # Bounding box center
            center_x = (x1 + x2) // 2

            # Create state for new person
            if track_id not in tracks:
                tracks[track_id] = new_track(center_x)

            state = tracks[track_id]

            state["last_seen"] = frame_number

            crop_x1 = max(0, min(width, x1))
            crop_y1 = max(0, min(height, y1))
            crop_x2 = max(0, min(width, x2))
            crop_y2 = max(0, min(height, y2))
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            # Classify each tracked person's crop periodically and reuse the
            # result between inferences to keep the camera loop responsive.
            if gender_model is not None and (
                frame_number - state["gender_last_frame"] >= GENDER_CLASSIFY_INTERVAL
            ):
                state["gender_last_frame"] = frame_number

                if person_crop.size:
                    try:
                        gender_result = gender_model.predict(
                            person_crop, verbose=False
                        )[0]
                        if gender_result.probs is not None:
                            gender_class_id = int(gender_result.probs.top1)
                            confidence = float(gender_result.probs.top1conf)
                        elif (
                            gender_result.boxes is not None
                            and len(gender_result.boxes) > 0
                        ):
                            best_box = int(gender_result.boxes.conf.argmax().item())
                            gender_class_id = int(
                                gender_result.boxes.cls[best_box].item()
                            )
                            confidence = float(
                                gender_result.boxes.conf[best_box].item()
                            )
                        else:
                            gender_class_id = None
                            confidence = 0.0

                        state["gender_label"] = (
                            gender_result.names[gender_class_id]
                            if gender_class_id is not None
                            and confidence >= GENDER_CONFIDENCE_THRESHOLD
                            else "uncertain"
                        )
                        state["gender_confidence"] = confidence
                    except Exception as exc:
                        logger.exception("Gender classification failed for track %s", track_id)

            if age_net is not None and (
                frame_number - state["age_last_frame"] >= AGE_CLASSIFY_INTERVAL
            ):
                state["age_last_frame"] = frame_number
                if person_crop.size:
                    try:
                        crop_height, crop_width = person_crop.shape[:2]
                        face_blob = cv2.dnn.blobFromImage(
                            person_crop,
                            1.0,
                            (300, 300),
                            (104.0, 177.0, 123.0),
                        )
                        face_net.setInput(face_blob)
                        face_detections = face_net.forward()

                        best_face = None
                        best_face_area = 0
                        for face_index in range(face_detections.shape[2]):
                            face_confidence = float(
                                face_detections[0, 0, face_index, 2]
                            )
                            if face_confidence < AGE_FACE_CONFIDENCE_THRESHOLD:
                                continue

                            face_box = (
                                face_detections[0, 0, face_index, 3:7]
                                * [crop_width, crop_height, crop_width, crop_height]
                            ).astype(int)
                            fx1, fy1, fx2, fy2 = face_box
                            fx1, fx2 = max(0, fx1), min(crop_width, fx2)
                            fy1, fy2 = max(0, fy1), min(crop_height, fy2)
                            face_width, face_height = fx2 - fx1, fy2 - fy1
                            face_area = face_width * face_height
                            if (
                                face_width >= AGE_MIN_FACE_SIZE
                                and face_height >= AGE_MIN_FACE_SIZE
                                and face_area > best_face_area
                            ):
                                best_face = person_crop[fy1:fy2, fx1:fx2]
                                best_face_area = face_area

                        if best_face is not None:
                            age_blob = cv2.dnn.blobFromImage(
                                best_face,
                                1.0,
                                (227, 227),
                                (78.4263377603, 87.7689143744, 114.895847746),
                                swapRB=False,
                            )
                            age_net.setInput(age_blob)
                            age_scores = age_net.forward()[0]
                            age_class = int(age_scores.argmax())
                            state["age_range"] = AGE_BUCKETS[age_class]
                            state["age_confidence"] = float(age_scores[age_class])
                        else:
                            state["age_range"] = "face not found"
                            state["age_confidence"] = 0.0
                    except Exception as exc:
                        logger.exception("Age estimation failed for track %s", track_id)

            person_label = (
                f"id:{track_id} {result.names[person_class_id]} {box_confidence:.0%}"
            )
            if gender_model is not None:
                gender_label = state["gender_label"] or "classifying..."
                confidence_text = (
                    f" {state['gender_confidence']:.0%}"
                    if state["gender_label"] not in (None, "uncertain")
                    else ""
                )
                person_label += f" | {gender_label}{confidence_text}"
            if age_net is not None:
                age_label = state["age_range"] or "finding face..."
                age_confidence_text = (
                    f" {state['age_confidence']:.0%}"
                    if state["age_confidence"] > 0
                    else ""
                )
                person_label += f" | age {age_label}{age_confidence_text}"
            log_track_update(track_id, state)
            annotator.box_label(
                (x1, y1, x2, y2),
                person_label,
                color=colors(person_class_id, True, result.colors),
            )

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
                    logger.info("IN: track id=%s", track_id)

                # RIGHT -> LEFT = OUT
                elif old_side == 1 and new_side == -1:
                    out_count += 1
                    logger.info("OUT: track id=%s", track_id)

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

    if gender_model is None:
        cv2.putText(
            output,
            "Gender model missing: add gender_classification.pt",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
        )

    if age_net is None:
        cv2.putText(
            output,
            "Age model missing: add files under face_detector/ and age_detector/",
            (10, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 255),
            1,
        )

    cv2.imshow("People Counter", output)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()

logger.info("Total IN: %s", in_count)
logger.info("Total OUT: %s", out_count)
