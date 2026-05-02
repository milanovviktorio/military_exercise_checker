import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pandas as pd

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("./Files/model.pkl")

try:
    feature_names = joblib.load("./Files/feature_names.pkl")
except FileNotFoundError:
    # Fallback: hardcoded in the same order as your CSV (minus Side & Label)
    feature_names = [
        "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
        "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
        "Knee_Ground_Angle", "Ankle_Ground_Angle"
    ]

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ── Helpers ───────────────────────────────────────────────────────────────────
def angle_between(a, b, c):
    """Angle at vertex b formed by a-b-c (degrees)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))


def ground_angle(a, b):
    """
    Angle of the segment a->b relative to horizontal ground (degrees, 0-90).
    """
    a, b = np.array(a), np.array(b)
    delta = b - a
    return np.degrees(np.arctan2(abs(delta[1]), abs(delta[0])))


def get_landmark_xy(lm, landmark):
    p = lm[landmark.value]
    return [p.x, p.y]


def extract_angles(landmarks, side="left"):
    """
    Returns a dict with keys matching your CSV feature columns exactly.
    side: 'right' or 'left'
    """
    lm = landmarks.landmark
    L  = mp_pose.PoseLandmark
    g  = get_landmark_xy

    if side == "right":
        SHOULDER = L.RIGHT_SHOULDER
        ELBOW    = L.RIGHT_ELBOW
        WRIST    = L.RIGHT_WRIST
        HIP      = L.RIGHT_HIP
        KNEE     = L.RIGHT_KNEE
        ANKLE    = L.RIGHT_ANKLE
        HEEL     = L.RIGHT_HEEL
    else:
        SHOULDER = L.LEFT_SHOULDER
        ELBOW    = L.LEFT_ELBOW
        WRIST    = L.LEFT_WRIST
        HIP      = L.LEFT_HIP
        KNEE     = L.LEFT_KNEE
        ANKLE    = L.LEFT_ANKLE
        HEEL     = L.LEFT_HEEL

    shoulder_pt = g(lm, SHOULDER)
    elbow_pt    = g(lm, ELBOW)
    wrist_pt    = g(lm, WRIST)
    hip_pt      = g(lm, HIP)
    knee_pt     = g(lm, KNEE)
    ankle_pt    = g(lm, ANKLE)
    heel_pt     = g(lm, HEEL)

    return {
        # ── Joint angles (3-point) ──────────────────────────────────────────
        "Shoulder_Angle":        angle_between(elbow_pt,    shoulder_pt, hip_pt),
        "Elbow_Angle":           angle_between(shoulder_pt, elbow_pt,    wrist_pt),
        "Hip_Angle":             angle_between(shoulder_pt, hip_pt,      knee_pt),
        "Knee_Angle":            angle_between(hip_pt,      knee_pt,     ankle_pt),
        "Ankle_Angle":           angle_between(knee_pt,     ankle_pt,    heel_pt),

        # ── Ground angles (segment vs horizontal) ───────────────────────────
        "Shoulder_Ground_Angle": ground_angle(shoulder_pt, elbow_pt),
        "Elbow_Ground_Angle":    ground_angle(elbow_pt,    wrist_pt),
        "Hip_Ground_Angle":      ground_angle(hip_pt,      knee_pt),
        "Knee_Ground_Angle":     ground_angle(knee_pt,     ankle_pt),
        "Ankle_Ground_Angle":    ground_angle(ankle_pt,    heel_pt),
    }


# ── Smoothing: rolling majority vote ─────────────────────────────────────────
WINDOW = 10
prediction_buffer = []

def smooth_prediction(new_pred):
    prediction_buffer.append(new_pred)
    if len(prediction_buffer) > WINDOW:
        prediction_buffer.pop(0)
    return max(set(prediction_buffer), key=prediction_buffer.count)


# ── Main loop ─────────────────────────────────────────────────────────────────
cap           = cv2.VideoCapture(0)
prev_time     = 0
current_label  = "Detecting..."
confidence_str = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame   = cv2.flip(frame, 1)
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            angles = extract_angles(results.pose_landmarks, side="left")
            X_live = pd.DataFrame([angles])[feature_names]

            pred  = model.predict(X_live)[0]
            proba = model.predict_proba(X_live).max()

            current_label  = smooth_prediction(pred)
            confidence_str = f"{proba * 100:.0f}%"

        except Exception as e:
            current_label  = "Error"
            confidence_str = str(e)[:50]

    # ── FPS ──────────────────────────────────────────────────────────────────
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # ── Overlay ───────────────────────────────────────────────────────────────
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"Exercise: {current_label}",
                (15, h - 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 180), 2)
    cv2.putText(frame, f"Confidence: {confidence_str}",
                (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {int(fps)}",
                (w - 110, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Exercise Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()