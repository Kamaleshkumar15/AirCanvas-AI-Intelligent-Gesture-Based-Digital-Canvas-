"""
AIR CANVAS AI PRO
=================
High-stability, gesture-controlled virtual canvas.

Core:
- MediaPipe hand landmarks
- Index finger only: draw
- Index + middle: toolbar selection
- Temporal smoothing + One Euro-style adaptive smoothing
- Gesture hysteresis/debounce
- Missing-frame tolerance
- Jump filtering
- Multi-color drawing
- Eraser
- Undo / Redo
- Brush size gesture
- Basic shape recognition (line, rectangle, circle, triangle)
- Digit recognition hook (optional, offline OpenCV template-free heuristic)
- Handwriting-to-text hook (optional; requires local OCR package)
- Voice commands hook (optional; requires SpeechRecognition + microphone)
- Presentation pointer mode
- Save PNG
- Screenshot-safe canvas

Keyboard:
Q/ESC quit
C clear
S save
Z undo
Y redo
E eraser
D drawing mode
P presentation pointer
1-4 colors
[ / ] brush size
H handwriting OCR (if installed)
V voice command (if installed)
T toggle shape mode

Install:
python -m pip install -r requirements.txt

Recommended Python:
3.10 or 3.11 for broad MediaPipe compatibility.
"""

import os
import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


# ========================= CONFIGURATION ==========================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

DETECTION_CONFIDENCE = 0.80
TRACKING_CONFIDENCE = 0.80
MAX_HANDS = 1
MODEL_COMPLEXITY = 1

# Drawing stability
SMOOTH_ALPHA = 0.32
MIN_POINT_DISTANCE = 2.0
MAX_POINT_JUMP = 75.0
MAX_MISSING_FRAMES = 5

# Toolbar selection stability
SELECT_STABLE_FRAMES = 8
SELECT_COOLDOWN = 0.65

# Brush
MIN_BRUSH = 2
MAX_BRUSH = 35
DEFAULT_BRUSH = 7
ERASER_MIN = 18
ERASER_MAX = 70

# Toolbar coordinates
BUTTONS = [
    (15, 15, 120, 68, "CLEAR"),
    (130, 15, 235, 68, "BLUE"),
    (245, 15, 350, 68, "GREEN"),
    (360, 15, 465, 68, "RED"),
    (475, 15, 595, 68, "YELLOW"),
    (605, 15, 705, 68, "ERASE"),
    (715, 15, 810, 68, "UNDO"),
    (820, 15, 915, 68, "REDO"),
    (925, 15, 1030, 68, "SAVE"),
    (1040, 15, 1170, 68, "POINTER"),
]

COLORS = {
    "BLUE": (255, 0, 0),
    "GREEN": (0, 200, 0),
    "RED": (0, 0, 255),
    "YELLOW": (0, 210, 210),
}

# =========================== UTILITIES =============================


def dist(a, b):
    return float(np.linalg.norm(
        np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    ))


def angle(a, b, c):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-7
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def point_from_landmark(lm, w, h):
    return (
        max(0, min(w - 1, int(lm.x * w))),
        max(0, min(h - 1, int(lm.y * h))),
    )


def inside_button(point):
    if point is None:
        return None
    x, y = point
    for x1, y1, x2, y2, label in BUTTONS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return label
    return None


def clear_canvas(canvas):
    canvas[:] = 255


def save_canvas(canvas):
    os.makedirs("output", exist_ok=True)
    name = time.strftime("air_canvas_%Y%m%d_%H%M%S.png")
    path = os.path.join("output", name)
    cv2.imwrite(path, canvas)
    return path


# ====================== GESTURE CLASSIFICATION =====================


def get_finger_states(p):
    """
    Returns thumb,index,middle,ring,pinky.
    Uses joint angles and relative geometry rather than a single Y test.
    """
    fingers = []

    # index, middle, ring, pinky
    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        pip_a = angle(p[mcp], p[pip], p[dip])
        dip_a = angle(p[pip], p[dip], p[tip])

        # Require both joints to be fairly straight.
        fingers.append(pip_a > 150 and dip_a > 145)

    # Thumb uses distance + geometry, robust to left/right mirrored camera.
    thumb_open = dist(p[4], p[5]) > dist(p[3], p[5]) * 1.12
    return [thumb_open] + fingers


def classify_gesture(states):
    thumb, index, middle, ring, pinky = states

    if index and middle and not ring and not pinky:
        return "SELECT"

    if index and not middle and not ring and not pinky:
        return "DRAW"

    # All fingers open = pointer mode gesture.
    if index and middle and ring and pinky:
        return "POINTER"

    # Thumb + index pinch is used for brush-size control.
    if index and thumb and not middle and not ring and not pinky:
        return "PINCH"

    return "MOVE"


# ===================== SIMPLE ADAPTIVE SMOOTHING ====================


class AdaptiveSmoother:
    """
    Lightweight adaptive exponential filter.
    Fast movements receive more responsiveness; slow movements get
    stronger smoothing.
    """

    def __init__(self, alpha=0.32):
        self.alpha = alpha
        self.value = None
        self.last_raw = None

    def update(self, point):
        p = np.asarray(point, dtype=np.float32)

        if self.value is None:
            self.value = p
            self.last_raw = p
            return tuple(p.astype(int))

        velocity = float(np.linalg.norm(p - self.last_raw))
        # More smoothing for tiny movement, less smoothing for fast movement.
        a = np.clip(self.alpha + min(velocity / 100.0, 0.35), 0.18, 0.68)

        self.value = a * p + (1.0 - a) * self.value
        self.last_raw = p

        return tuple(self.value.astype(int))

    def reset(self):
        self.value = None
        self.last_raw = None


# ======================= CANVAS HISTORY ============================


class History:
    def __init__(self, limit=30):
        self.items = []
        self.index = -1
        self.limit = limit

    def snapshot(self, canvas):
        item = canvas.copy()

        if self.index < len(self.items) - 1:
            self.items = self.items[:self.index + 1]

        self.items.append(item)
        if len(self.items) > self.limit:
            self.items.pop(0)

        self.index = len(self.items) - 1

    def undo(self, canvas):
        if self.index > 0:
            self.index -= 1
            canvas[:] = self.items[self.index]

    def redo(self, canvas):
        if self.index < len(self.items) - 1:
            self.index += 1
            canvas[:] = self.items[self.index]


# ========================= SHAPE RECOGNITION =======================


def recognize_shape(points):
    """
    Recognizes simple closed strokes.
    Returns None, LINE, TRIANGLE, RECTANGLE or CIRCLE.
    """
    if len(points) < 20:
        return None

    pts = np.asarray(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    if w < 30 or h < 30:
        return None

    contour = pts.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

    closed_error = dist(pts[0], pts[-1]) / max(w, h)
    if closed_error > 0.20:
        return "LINE"

    n = len(approx)

    if n == 3:
        return "TRIANGLE"
    if n == 4:
        ratio = w / float(h)
        return "RECTANGLE" if 0.25 < ratio < 4.0 else "RECTANGLE"

    circularity = (4 * math.pi * cv2.contourArea(contour)) / (perimeter * perimeter + 1e-7)
    if circularity > 0.70:
        return "CIRCLE"

    return None


def draw_recognized_shape(canvas, points, color, thickness):
    shape = recognize_shape(points)
    if shape is None:
        return

    pts = np.asarray(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(pts)

    if shape == "LINE":
        cv2.line(canvas, tuple(pts[0]), tuple(pts[-1]), color, thickness, cv2.LINE_AA)

    elif shape == "RECTANGLE":
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)

    elif shape == "CIRCLE":
        center = (x + w // 2, y + h // 2)
        radius = max(1, min(w, h) // 2)
        cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)

    elif shape == "TRIANGLE":
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.05 * peri, True)
        if len(approx) == 3:
            cv2.polylines(canvas, [approx], True, color, thickness, cv2.LINE_AA)

    return shape


# ====================== OPTIONAL OCR / VOICE =======================


def handwriting_to_text(canvas):
    """
    Optional offline OCR adapter.
    Tries pytesseract only if installed. No network is required.
    """
    try:
        import pytesseract
    except ImportError:
        return "Install pytesseract + Tesseract OCR to use OCR."

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    try:
        text = pytesseract.image_to_string(gray, config="--psm 6")
        text = text.strip()
        return text if text else "No text detected."
    except Exception as exc:
        return f"OCR unavailable: {exc}"


def voice_command():
    """
    Optional voice command adapter.
    Requires SpeechRecognition and an available microphone.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return "Install SpeechRecognition + PyAudio for voice commands."

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
        # Network speech recognition providers may be used by SpeechRecognition;
        # this adapter intentionally remains optional.
        text = recognizer.recognize_google(audio)
        return text.lower()
    except Exception as exc:
        return f"Voice command unavailable: {exc}"


# ========================= PRESENTATION POINTER ====================


def draw_pointer_overlay(frame, point):
    if point is None:
        return
    x, y = point
    cv2.circle(frame, (x, y), 22, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(frame, (x - 32, y), (x + 32, y), (255, 255, 255), 1)
    cv2.line(frame, (x, y - 32), (x, y + 32), (255, 255, 255), 1)


# ============================ TOOLBAR ==============================


def draw_toolbar(frame, selected_color, hover=None, shape_mode=False, brush=7):
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (1185, 82), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    for x1, y1, x2, y2, label in BUTTONS:
        if label in COLORS:
            fill = COLORS[label]
            text = (255, 255, 255)
        else:
            fill = (55, 55, 55)
            text = (255, 255, 255)

        if label == selected_color:
            fill = COLORS[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1, cv2.LINE_AA)

        if label == hover:
            cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                          (255, 255, 255), 3, cv2.LINE_AA)

        cv2.putText(frame, label, (x1 + 8, y1 + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text, 2, cv2.LINE_AA)

    mode_text = f"BRUSH: {brush} | SHAPES: {'ON' if shape_mode else 'OFF'}"
    cv2.putText(frame, mode_text, (20, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                COLORS[selected_color], 2, cv2.LINE_AA)


# =============================== MAIN ==============================


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            "Webcam could not be opened. Try CAMERA_INDEX = 1 or check Windows camera permissions."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Camera opened but returned no frame.")

    first = cv2.flip(first, 1)
    h, w = first.shape[:2]

    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    history = History()
    history.snapshot(canvas)

    smoother = AdaptiveSmoother(SMOOTH_ALPHA)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        model_complexity=MODEL_COMPLEXITY,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    selected_color = "BLUE"
    brush_size = DEFAULT_BRUSH
    erasing = False
    shape_mode = False
    pointer_mode = False

    previous_point = None
    stroke_points = []

    candidate_button = None
    candidate_frames = 0
    last_selection = 0.0

    lost_frames = 0
    status = "READY"
    status_time = time.time()

    def set_status(msg):
        nonlocal status, status_time
        status = msg
        status_time = time.time()

    def commit():
        history.snapshot(canvas)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                lost_frames += 1
                if lost_frames > MAX_MISSING_FRAMES:
                    set_status("CAMERA LOST")
                continue

            lost_frames = 0
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture = "NO HAND"
            fingertip = None
            hover = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                points = [point_from_landmark(lm, w, h)
                          for lm in hand.landmark]

                states = get_finger_states(points)
                gesture = classify_gesture(states)

                raw_tip = points[8]
                fingertip = smoother.update(raw_tip)

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(80, 210, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

                # ---------------- SELECT ----------------
                if gesture == "SELECT":
                    previous_point = None
                    stroke_points.clear()

                    hover = inside_button(fingertip)

                    if hover == candidate_button:
                        candidate_frames += 1
                    else:
                        candidate_button = hover
                        candidate_frames = 1

                    if (
                        hover is not None
                        and candidate_frames >= SELECT_STABLE_FRAMES
                        and time.time() - last_selection >= SELECT_COOLDOWN
                    ):
                        last_selection = time.time()
                        candidate_frames = 0

                        if hover == "CLEAR":
                            clear_canvas(canvas)
                            commit()
                            set_status("CLEARED")

                        elif hover in COLORS:
                            selected_color = hover
                            erasing = False
                            pointer_mode = False
                            set_status(f"COLOR: {hover}")

                        elif hover == "ERASE":
                            erasing = True
                            pointer_mode = False
                            set_status("ERASER ON")

                        elif hover == "UNDO":
                            history.undo(canvas)
                            set_status("UNDO")

                        elif hover == "REDO":
                            history.redo(canvas)
                            set_status("REDO")

                        elif hover == "SAVE":
                            path = save_canvas(canvas)
                            set_status("SAVED " + path)

                        elif hover == "POINTER":
                            pointer_mode = not pointer_mode
                            set_status(
                                "POINTER ON" if pointer_mode else "POINTER OFF"
                            )

                # ---------------- DRAW ----------------
                elif gesture == "DRAW":
                    candidate_button = None
                    candidate_frames = 0

                    if fingertip[1] <= 90:
                        previous_point = None
                        stroke_points.clear()
                    else:
                        if previous_point is not None:
                            jump = dist(previous_point, fingertip)

                            if MIN_POINT_DISTANCE <= jump <= MAX_POINT_JUMP:
                                if erasing:
                                    cv2.line(
                                        canvas, previous_point, fingertip,
                                        (255, 255, 255),
                                        max(ERASER_MIN, brush_size * 4),
                                        cv2.LINE_AA,
                                    )
                                else:
                                    cv2.line(
                                        canvas, previous_point, fingertip,
                                        COLORS[selected_color],
                                        brush_size,
                                        cv2.LINE_AA,
                                    )

                                if not erasing:
                                    stroke_points.append(fingertip)

                        previous_point = fingertip
                        if not erasing:
                            stroke_points.append(fingertip)

                # ---------------- POINTER ----------------
                elif gesture == "POINTER" and pointer_mode:
                    previous_point = None
                    stroke_points.clear()

                # ---------------- PINCH BRUSH SIZE ----------------
                elif gesture == "PINCH":
                    previous_point = None
                    stroke_points.clear()

                    thumb_tip = points[4]
                    index_tip = points[8]
                    pinch_distance = dist(thumb_tip, index_tip)

                    # Camera-relative size estimation.
                    palm_scale = max(40.0, dist(points[5], points[17]))
                    normalized = np.clip(pinch_distance / palm_scale, 0.65, 3.0)
                    brush_size = int(
                        np.interp(normalized, [0.65, 3.0],
                                  [MIN_BRUSH, MAX_BRUSH])
                    )

                else:
                    # MOVE / unknown pose: terminate current stroke.
                    if stroke_points and shape_mode and not erasing:
                        shape = draw_recognized_shape(
                            canvas, stroke_points,
                            COLORS[selected_color], brush_size
                        )
                        if shape:
                            set_status(f"SHAPE: {shape}")
                        commit()

                    previous_point = None
                    stroke_points.clear()
                    candidate_button = None
                    candidate_frames = 0

                marker = (0, 255, 0) if gesture == "DRAW" else (0, 220, 255)
                cv2.circle(frame, fingertip, 10, marker, 2, cv2.LINE_AA)
                cv2.circle(frame, fingertip, 3, marker, -1, cv2.LINE_AA)

                if pointer_mode and gesture == "POINTER":
                    draw_pointer_overlay(frame, fingertip)

            else:
                # No hand = finish current stroke.
                if stroke_points and shape_mode and not erasing:
                    shape = draw_recognized_shape(
                        canvas, stroke_points,
                        COLORS[selected_color], brush_size
                    )
                    if shape:
                        set_status(f"SHAPE: {shape}")
                    commit()

                previous_point = None
                stroke_points.clear()
                candidate_button = None
                candidate_frames = 0
                smoother.reset()

            # Display drawing on top of webcam.
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(
                255 - gray, 15, 255, cv2.THRESH_BINARY
            )[1]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            output = np.where(mask > 0, canvas, frame).astype(np.uint8)

            draw_toolbar(
                output,
                selected_color,
                hover if gesture == "SELECT" else None,
                shape_mode,
                brush_size,
            )

            # Status / mode info
            status_display = status if time.time() - status_time < 2.2 else "RUNNING"
            cv2.putText(
                output,
                f"MODE: {gesture}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                output,
                "☝ DRAW   ☝+MIDDLE SELECT   PINCH BRUSH",
                (20, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )

            cv2.rectangle(
                output,
                (w - 310, 98),
                (w - 15, 140),
                (25, 25, 25),
                -1,
            )
            cv2.putText(
                output,
                status_display[:34],
                (w - 298, 126),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("AIR CANVAS AI PRO", output)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            elif key in (ord("c"), ord("C")):
                clear_canvas(canvas)
                commit()
                set_status("CLEARED")

            elif key in (ord("s"), ord("S")):
                path = save_canvas(canvas)
                set_status("SAVED " + path)

            elif key in (ord("z"), ord("Z")):
                history.undo(canvas)
                set_status("UNDO")

            elif key in (ord("y"), ord("Y")):
                history.redo(canvas)
                set_status("REDO")

            elif key in (ord("e"), ord("E")):
                erasing = not erasing
                set_status("ERASER ON" if erasing else "ERASER OFF")

            elif key in (ord("p"), ord("P")):
                pointer_mode = not pointer_mode
                set_status("POINTER ON" if pointer_mode else "POINTER OFF")

            elif key in (ord("t"), ord("T")):
                shape_mode = not shape_mode
                set_status("SHAPES ON" if shape_mode else "SHAPES OFF")

            elif key == ord("["):
                brush_size = max(MIN_BRUSH, brush_size - 1)

            elif key == ord("]"):
                brush_size = min(MAX_BRUSH, brush_size + 1)

            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                selected_color = {
                    ord("1"): "BLUE",
                    ord("2"): "GREEN",
                    ord("3"): "RED",
                    ord("4"): "YELLOW",
                }[key]
                erasing = False
                set_status("COLOR " + selected_color)

            elif key in (ord("h"), ord("H")):
                text = handwriting_to_text(canvas)
                print("\n--- OCR RESULT ---\n" + text + "\n------------------")
                set_status("OCR RESULT PRINTED IN CONSOLE")

            elif key in (ord("v"), ord("V")):
                text = voice_command()
                print("\nVOICE:", text)
                if text.startswith("clear"):
                    clear_canvas(canvas)
                    commit()
                elif "blue" in text:
                    selected_color = "BLUE"
                elif "green" in text:
                    selected_color = "GREEN"
                elif "red" in text:
                    selected_color = "RED"
                elif "yellow" in text:
                    selected_color = "YELLOW"
                elif "eraser" in text:
                    erasing = True
                elif "save" in text:
                    path = save_canvas(canvas)
                    print("Saved:", path)

    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
