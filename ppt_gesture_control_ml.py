import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
import joblib
import pyautogui

MODEL_PATH = "gesture_model.joblib"

# MediaPipe modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# --- Smoothing settings ---
SMOOTH_WINDOW = 9          # number of recent predictions to vote on
MIN_STABLE_COUNT = 6       # must appear this many times in window to accept
PROB_THRESHOLD = 0.75      # ML confidence threshold
DEBOUNCE_SEC = 0.8         # delay between actions

# Map gesture label -> PowerPoint key
GESTURE_TO_KEY = {
    "NEXT": "right",
    "PREV": "left",
    "START": "f5",
    "EXIT": "esc",
    "NONE": None
}


def landmarks_to_feature_vec(hand_landmarks):
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten()


def majority_vote(pred_queue):
    c = Counter(pred_queue)
    label, count = c.most_common(1)[0]
    return label, count


def main():
    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Camera not opened.")
        return

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    face_det = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    pred_hist = deque(maxlen=SMOOTH_WINDOW)
    last_action_time = 0.0
    last_action_label = None

    print("=== PPT Gesture Control (ML + Smoothing + Face Detection) ===")
    print("Open PowerPoint in slideshow mode (or use START gesture).")
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- Face detection ----
        face_res = face_det.process(rgb)
        face_found = bool(face_res.detections)

        if face_res.detections:
            for det in face_res.detections:
                mp_draw.draw_detection(frame, det)

        # ---- Hand detection ----
        res = hands.process(rgb)

        live_label = "NO_HAND"
        live_prob = 0.0

        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            feat = landmarks_to_feature_vec(hl).reshape(1, -1)

            probs = model.predict_proba(feat)[0]
            classes = model.classes_
            idx = int(np.argmax(probs))
            pred = classes[idx]
            prob = float(probs[idx])

            if prob >= PROB_THRESHOLD:
                pred_hist.append(pred)
                live_label, live_prob = pred, prob
            else:
                pred_hist.append("NONE")
                live_label, live_prob = "NONE", prob
        else:
            pred_hist.append("NONE")

        # ---- Smoothing (majority vote) ----
        stable_label, stable_count = majority_vote(pred_hist)
        action_label = stable_label if stable_count >= MIN_STABLE_COUNT else "NONE"
        action_key = GESTURE_TO_KEY.get(action_label, None)

        # Optional safety gate: allow slide control only if a face is detected
        # Uncomment the next 2 lines if you want this behavior:
        # if not face_found:
        #     action_key, action_label = None, "NONE"

        # ---- Debounce + key press ----
        now = time.time()
        can_fire = (now - last_action_time) >= DEBOUNCE_SEC

        if action_key is not None and can_fire:
            if action_label != last_action_label or (now - last_action_time) >= DEBOUNCE_SEC:
                pyautogui.press(action_key)
                last_action_time = now
                last_action_label = action_label

        # ---- UI overlay ----
        cv2.putText(frame, f"Face: {'YES' if face_found else 'NO'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Live: {live_label} ({live_prob:.2f})", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Stable: {stable_label} [{stable_count}/{SMOOTH_WINDOW}]", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Action: {action_label}", (10, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(frame, "q = quit", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("PPT Gesture Control", frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()