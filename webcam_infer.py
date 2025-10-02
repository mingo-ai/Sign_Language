# # webcam_infer.py
# # Run: python webcam_infer.py

# import os
# os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# os.environ.setdefault("MKL_NUM_THREADS", "1")
# os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# import time
# import cv2
# import joblib
# import mediapipe as mp
# import pyttsx3
# import threading
# from pathlib import Path

# # ---------- Config ----------
# MODEL_PATH = "gesture_model.pkl"  # trained RandomForest model on 42 features (x,y)
# CAM_INDEX = 0
# MIN_DET_CONF = 0.7
# MAX_HANDS = 1

# # TTS policy
# MIN_INTERVAL_SEC = 2.0     # don't speak more frequently than every 2 seconds
# REQUIRE_CHANGE = True      # only speak when the (translated) word changed
# # ----------------------------

# # ---------- (Optional) translation hook ----------
# def translate_text(text: str) -> str:
#     """
#     Replace this stub with your translator if you use one.
#     For now it returns the original text (English).
#     """
#     return text

# # --------- TTS engine ---------
# engine = pyttsx3.init()
# engine.setProperty('rate', 150)

# def say_async(text: str):
#     threading.Thread(
#         target=lambda: (engine.say(text), engine.runAndWait()),
#         daemon=True
#     ).start()

# # --------- Load model ---------
# if not Path(MODEL_PATH).exists():
#     raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
# clf = joblib.load(MODEL_PATH)
# print(f"Loaded model: {MODEL_PATH}")

# # --------- MediaPipe Hands ---------
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=MAX_HANDS,
#     min_detection_confidence=MIN_DET_CONF
# )

# # Pure-OpenCV drawing (no matplotlib/pillow)
# def draw_hand(frame, hand_landmarks, connections):
#     h, w, _ = frame.shape
#     pts = []
#     for lm in hand_landmarks.landmark:
#         x, y = int(lm.x * w), int(lm.y * h)
#         pts.append((x, y))
#     for (x, y) in pts:
#         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#     if connections:
#         for a, b in connections:
#             cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)

# # --------- Webcam ---------
# cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
# if not cap.isOpened():
#     raise RuntimeError("Webcam not accessible.")

# print("Webcam running. Press 'q' to quit, 'c' to clear sentence.")

# spoken_words = []
# last_word_for_tts = None       # last spoken (translated) word
# last_tts_time = 0.0

# try:
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             print("Failed to grab frame.")
#             break

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         res = hands.process(rgb)

#         if res.multi_hand_landmarks:
#             for hlm in res.multi_hand_landmarks:
#                 # Model trained on x,y only (42 features)
#                 lm = [[p.x, p.y] for p in hlm.landmark]
#                 if len(lm) == 21:
#                     flat = [v for pt in lm for v in pt]
#                     pred = str(clf.predict([flat])[0])

#                     # Draw
#                     draw_hand(frame, hlm, mp_hands.HAND_CONNECTIONS)
#                     cv2.putText(frame, pred, (10, 40),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

#                     # Build sentence display
#                     if not spoken_words:
#                         spoken_words.append(pred.capitalize())
#                     else:
#                         # Show repeated words in the sentence too (visual history)
#                         if pred.lower() != spoken_words[-1].lower():
#                             spoken_words.append(pred.lower())

#                     # ---- TTS gating logic ----
#                     now = time.time()
#                     to_speak = translate_text(pred)   # if you add translation, change happens here
#                     changed = (to_speak != last_word_for_tts)

#                     # Speak only if (changed if required) AND interval passed
#                     if (not REQUIRE_CHANGE or changed) and (now - last_tts_time >= MIN_INTERVAL_SEC):
#                         say_async(to_speak)
#                         last_word_for_tts = to_speak
#                         last_tts_time = now

#         # Overlay sentence
#         sentence = " ".join(spoken_words)
#         h, w, _ = frame.shape
#         (tw, th), _ = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
#         cv2.rectangle(frame, (10, h - 60), (10 + tw + 10, h - 20), (0, 0, 0), -1)
#         cv2.putText(frame, sentence, (15, h - 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#         cv2.imshow("ASL Webcam with TTS (q=quit, c=clear)", frame)

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('c'):
#             spoken_words.clear()
#             print("Sentence cleared.")

# finally:
#     cap.release()
#     cv2.destroyAllWindows()
#     try:
#         engine.stop()
#     except Exception:
#         pass







# webcam_infer.py
# Run: python webcam_infer.py

# webcam_infer_tts_stable.py
# Run: python webcam_infer_tts_stable.py

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from pathlib import Path
import threading
import cv2
import joblib
import mediapipe as mp
import pyttsx3

# ---------------- Config ----------------
MODEL_PATH = "gesture_model.pkl"  # RF trained on 42 features (x,y)
CAM_INDEX = 0
MIN_DET_CONF = 0.7
MAX_HANDS = 1

STABLE_SEC = 1.5        # must hold the SAME label this long before speaking
MIN_INTERVAL_SEC = 2.0  # minimum time between two spoken outputs
# ----------------------------------------

# ---- Optional translation hook (no-op) ----
def translate_text(text: str) -> str:
    return text  # replace with your translator if needed

# ---- TTS (with busy flag) ----
engine = pyttsx3.init()
engine.setProperty('rate', 150)
_tts_busy = False
_tts_lock = threading.Lock()

def _say_worker(text: str):
    global _tts_busy
    try:
        engine.say(text)
        engine.runAndWait()
    finally:
        with _tts_lock:
            _tts_busy = False

def say(text: str):
    """Speak synchronously from the app's perspective (we gate new accepts while busy)."""
    global _tts_busy
    with _tts_lock:
        if _tts_busy:
            return
        _tts_busy = True
    t = threading.Thread(target=_say_worker, args=(text,), daemon=True)
    t.start()

# ---- Load model ----
if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
clf = joblib.load(MODEL_PATH)
print(f"Loaded model: {MODEL_PATH}")

# ---- MediaPipe Hands ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=MAX_HANDS,
    min_detection_confidence=MIN_DET_CONF
)

# ---- Pure OpenCV drawing (no matplotlib/pillow) ----
def draw_hand(frame, hand_landmarks, connections):
    h, w, _ = frame.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    if connections:
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 1)

# ---- Webcam ----
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible.")
print("Webcam running. Press 'q' to quit, 'c' to clear sentence.")

# ---- State for stability + speech gating ----
spoken_words = []

candidate_label = None        # label currently being considered
candidate_since = 0.0         # wall-clock time when candidate was first seen
last_spoken_label = None      # last label we actually spoke (after translation)
last_spoken_time = 0.0        # wall-clock time of last speech

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        pred_label = None
        if res.multi_hand_landmarks:
            for hlm in res.multi_hand_landmarks:
                # Your model expects 42 features (x,y). If you trained on 63, include z as well.
                vec = [[p.x, p.y] for p in hlm.landmark]  # 21x2
                if len(vec) == 21:
                    flat = [v for pt in vec for v in pt]  # len=42
                    pred_label = str(clf.predict([flat])[0])

                    draw_hand(frame, hlm, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, pred_label, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                break  # we only use the first detected hand for decision timing

        now = time.time()

        # ----- Stability logic -----
        if pred_label is None or (_tts_busy):
            # If no hand OR TTS is currently speaking, don't accumulate stability time.
            candidate_label = None
            candidate_since = 0.0
        else:
            if pred_label != candidate_label:
                # New candidate starts timing now
                candidate_label = pred_label
                candidate_since = now
            # else: same candidate continues

            # When candidate has been stable long enough:
            if candidate_label is not None and (now - candidate_since) >= STABLE_SEC:
                # Prepare what we would speak (e.g., translated)
                to_speak = translate_text(candidate_label)

                # Check min interval and content change (vs the last actually spoken)
                interval_ok = (now - last_spoken_time) >= MIN_INTERVAL_SEC
                changed_ok = (to_speak != last_spoken_label)

                if interval_ok and changed_ok and not _tts_busy:
                    say(to_speak)                 # start speaking
                    last_spoken_label = to_speak  # store what we actually spoke
                    last_spoken_time = time.time()

                    # Update sentence (capitalize first word)
                    if not spoken_words:
                        spoken_words.append(candidate_label.capitalize())
                    else:
                        spoken_words.append(candidate_label.lower())

                    # Reset stability so we require another 1.5s for the next word
                    candidate_label = None
                    candidate_since = 0.0

        # ----- Overlay sentence -----
        sentence = " ".join(spoken_words)
        h, w, _ = frame.shape
        (tw, th), _ = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (10, h - 60), (10 + tw + 10, h - 20), (0, 0, 0), -1)
        cv2.putText(frame, sentence, (15, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Small UI: show stability progress bar
        if candidate_label and candidate_since > 0:
            prog = min(1.0, (now - candidate_since) / STABLE_SEC)
            bar_w = int(200 * prog)
            cv2.rectangle(frame, (10, h - 80), (10 + 200, h - 70), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, h - 80), (10 + bar_w, h - 70), (0, 200, 0), -1)
            cv2.putText(frame, f"Stable: {candidate_label} ({int(prog*100)}%)",
                        (220, h - 72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("ASL Webcam (stable TTS)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            spoken_words.clear()
            candidate_label = None
            candidate_since = 0.0
            print("Sentence cleared.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    try:
        engine.stop()
    except Exception:
        pass
