import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pandas as pd
import sqlite3
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  IDEAL ANGLE RANGES  (min, max) per exercise per joint
# ══════════════════════════════════════════════════════════════════════════════
IDEAL_RANGES = {
    "Push Ups": {
        "Elbow_Angle": (100, 180),
    },
    "Pull ups": {
        "Elbow_Angle": (60, 180),
    },
    "Squats": {
        "Knee_Angle":  (80, 100),
        "Hip_Angle":   (80, 100),
        "Ankle_Angle": (70,  90),
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  REP COUNTING CONFIG  — which joint drives the rep + stage thresholds
#  ideal_span: the expected full range of motion in degrees for a good rep
# ══════════════════════════════════════════════════════════════════════════════
REP_CONFIG = {
    "Push Ups": {
        "joint":       "Elbow_Angle",
        "down_thresh": 110,   # angle below this → "down" position
        "up_thresh":   160,   # angle above this → "up" position
        "ideal_span":   80,   # 180 - 100 from IDEAL_RANGES
    },
    "Pull ups": {
        "joint":       "Elbow_Angle",
        "down_thresh":  90,   # fully curled = small angle
        "up_thresh":   150,   # arms extended = large angle
        "ideal_span":  120,   # 180 - 60
    },
    "Squats": {
        "joint":       "Knee_Angle",
        "down_thresh": 100,   # knees bent
        "up_thresh":   160,   # standing
        "ideal_span":   80,   # 160 - 80 (uses up_thresh - down_thresh as proxy)
    },
}

# Joint display names
JOINT_LABELS = {
    "Shoulder_Angle":        "Shoulder",
    "Elbow_Angle":           "Elbow",
    "Hip_Angle":             "Hip",
    "Knee_Angle":            "Knee",
    "Ankle_Angle":           "Ankle",
    "Shoulder_Ground_Angle": "Shldr(gnd)",
    "Elbow_Ground_Angle":    "Elbow(gnd)",
    "Hip_Ground_Angle":      "Hip(gnd)",
    "Knee_Ground_Angle":     "Knee(gnd)",
    "Ankle_Ground_Angle":    "Ankle(gnd)",
}

# Which MediaPipe landmark is the vertex for each angle
JOINT_VERTEX = {
    "Shoulder_Angle": "SHOULDER",
    "Elbow_Angle":    "ELBOW",
    "Hip_Angle":      "HIP",
    "Knee_Angle":     "KNEE",
    "Ankle_Angle":    "ANKLE",
}

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def init_db(path="workout_tracker.db"):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            date         TEXT,
            exercise     TEXT,
            total_reps   INTEGER,
            avg_rom_pct  REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS reps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER,
            rep_number  INTEGER,
            exercise    TEXT,
            joint       TEXT,
            min_angle   REAL,
            max_angle   REAL,
            angle_span  REAL,
            ideal_span  REAL,
            rom_percent REAL,
            timestamp   TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    return conn

def save_rep_to_db(conn, session_id, rep_number, exercise, joint,
                   min_a, max_a, span, ideal_span, rom_pct):
    conn.execute("""
        INSERT INTO reps
        (session_id, rep_number, exercise, joint, min_angle, max_angle,
         angle_span, ideal_span, rom_percent, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (session_id, rep_number, exercise, joint,
          round(min_a, 1), round(max_a, 1), round(span, 1),
          ideal_span, round(rom_pct, 1),
          datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()

def save_session_to_db(conn, exercise, total_reps, roms):
    avg = sum(roms) / len(roms) if roms else 0.0
    cur = conn.execute("""
        INSERT INTO sessions (date, exercise, total_reps, avg_rom_pct)
        VALUES (?,?,?,?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), exercise, total_reps, avg))
    conn.commit()
    return cur.lastrowid

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("./Files/model.pkl")

try:
    feature_names = joblib.load("./Files/feature_names.pkl")
except FileNotFoundError:
    feature_names = [
        "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
        "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
        "Knee_Ground_Angle", "Ankle_Ground_Angle",
    ]

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# ── Angle helpers ─────────────────────────────────────────────────────────────
def angle_between(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

def ground_angle(a, b):
    a, b  = np.array(a), np.array(b)
    delta = b - a
    return float(np.degrees(np.arctan2(abs(delta[1]), abs(delta[0]))))

def get_xy(lm, landmark):
    p = lm[landmark.value]
    return [p.x, p.y]

def get_px(lm, landmark, w, h):
    """Return pixel (x, y) for a landmark."""
    p = lm[landmark.value]
    return int(p.x * w), int(p.y * h)

def extract_angles(landmarks, side="left"):
    lm = landmarks.landmark
    L  = mp_pose.PoseLandmark
    g  = get_xy

    if side == "right":
        SHOULDER, ELBOW, WRIST = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST
        HIP, KNEE, ANKLE, HEEL = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL
    else:
        SHOULDER, ELBOW, WRIST = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
        HIP, KNEE, ANKLE, HEEL = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL

    sh = g(lm, SHOULDER); el = g(lm, ELBOW);   wr = g(lm, WRIST)
    hi = g(lm, HIP);      kn = g(lm, KNEE);    an = g(lm, ANKLE); he = g(lm, HEEL)

    return {
        "Shoulder_Angle":        angle_between(el, sh, hi),
        "Elbow_Angle":           angle_between(sh, el, wr),
        "Hip_Angle":             angle_between(sh, hi, kn),
        "Knee_Angle":            angle_between(hi, kn, an),
        "Ankle_Angle":           angle_between(kn, an, he),
        "Shoulder_Ground_Angle": ground_angle(sh, el),
        "Elbow_Ground_Angle":    ground_angle(el, wr),
        "Hip_Ground_Angle":      ground_angle(hi, kn),
        "Knee_Ground_Angle":     ground_angle(kn, an),
        "Ankle_Ground_Angle":    ground_angle(an, he),
    }

# ── Draw angle label near a joint ─────────────────────────────────────────────
def draw_angle_near_joint(frame, angle_deg, px, py, highlight=False):
    """Draw the angle value in a small pill next to the joint landmark."""
    text  = f"{angle_deg:.0f}{chr(176)}"   # e.g.  "142°"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad   = 4
    ox, oy = px + 10, py - 10          # offset so label doesn't cover the dot

    bg    = (40, 40, 0)    if highlight else (20, 20, 20)
    color = (0, 220, 255)  if highlight else (0, 220, 220)

    cv2.rectangle(frame,
                  (ox - pad, oy - th - pad),
                  (ox + tw + pad, oy + pad),
                  bg, -1)
    cv2.putText(frame, text, (ox, oy), font, scale, color, thick, cv2.LINE_AA)

def draw_all_joint_angles(frame, landmarks, angles, w, h, side="left"):
    """
    For every angle we compute, find its vertex landmark and draw the
    angle value right next to it on screen.
    """
    lm = landmarks.landmark
    L  = mp_pose.PoseLandmark

    prefix = "LEFT_" if side == "left" else "RIGHT_"

    landmark_for = {
        "Shoulder_Angle": prefix + "SHOULDER",
        "Elbow_Angle":    prefix + "ELBOW",
        "Hip_Angle":      prefix + "HIP",
        "Knee_Angle":     prefix + "KNEE",
        "Ankle_Angle":    prefix + "ANKLE",
        # ground angles: draw at the proximal landmark
        "Shoulder_Ground_Angle": prefix + "SHOULDER",
        "Elbow_Ground_Angle":    prefix + "ELBOW",
        "Hip_Ground_Angle":      prefix + "HIP",
        "Knee_Ground_Angle":     prefix + "KNEE",
        "Ankle_Ground_Angle":    prefix + "ANKLE",
    }

    drawn_at = {}   # avoid stacking two labels on the exact same landmark

    for joint_key, angle_val in angles.items():
        lm_name = landmark_for.get(joint_key)
        if lm_name is None:
            continue
        try:
            lm_enum = getattr(L, lm_name)
        except AttributeError:
            continue

        p = lm[lm_enum.value]
        if p.visibility < 0.5:
            continue

        px, py = int(p.x * w), int(p.y * h)

        # If another angle already placed a label here, nudge downward
        key = (px // 5, py // 5)      # bucket to nearby pixels
        offset_y = drawn_at.get(key, 0)
        drawn_at[key] = offset_y + 18

        draw_angle_near_joint(frame, angle_val, px, py + offset_y)

# ── ROM % calculation ─────────────────────────────────────────────────────────
def rom_percent(angle, min_angle, max_angle):
    span = max_angle - min_angle
    if span == 0:
        return 100.0, True
    pct      = (angle - min_angle) / span * 100.0
    pct      = max(0.0, min(100.0, pct))
    in_range = min_angle <= angle <= max_angle
    return pct, in_range

# ── ROM bar drawing ───────────────────────────────────────────────────────────
BAR_W   = 180
BAR_H   = 16
BAR_X   = 10
BAR_GAP = 30

def draw_rom_bars(frame, angles, exercise):
    ranges = IDEAL_RANGES.get(exercise, {})
    if not ranges:
        return

    n_bars  = len(ranges)
    panel_h = n_bars * BAR_GAP + 20
    panel_w = BAR_X * 2 + BAR_W + 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 22
    for joint, (lo, hi) in ranges.items():
        angle    = angles.get(joint, 0.0)
        pct, ok  = rom_percent(angle, lo, hi)
        label    = JOINT_LABELS.get(joint, joint)

        bar_color  = (0, 200, 80)    if ok else (0, 80, 220)
        text_color = (200, 255, 200) if ok else (100, 180, 255)

        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (60, 60, 60), -1)
        fill_w = int(BAR_W * pct / 100.0)
        if fill_w > 0:
            cv2.rectangle(frame, (BAR_X, y), (BAR_X + fill_w, y + BAR_H), bar_color, -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (120, 120, 120), 1)

        text = f"{label}: {angle:.0f}° ({pct:.0f}%)"
        cv2.putText(frame, text, (BAR_X + BAR_W + 6, y + BAR_H - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)
        y += BAR_GAP

# ── Rep ROM% banner ───────────────────────────────────────────────────────────
def draw_rep_rom_banner(frame, rom_pct, rep_num, w, h):
    """Large centred banner shown for a couple of seconds after each rep."""
    if rom_pct >= 90:
        quality, color = "FULL ROM", (0, 220, 100)
    elif rom_pct >= 70:
        quality, color = "GOOD ROM", (0, 200, 255)
    elif rom_pct >= 50:
        quality, color = "PARTIAL ROM", (0, 140, 255)
    else:
        quality, color = "POOR ROM", (50, 50, 230)

    cx, cy = w // 2, h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - 170, cy - 60), (cx + 170, cy + 50), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    line1 = f"Rep {rep_num}:  {rom_pct:.0f}% ROM"
    line2 = quality
    font  = cv2.FONT_HERSHEY_SIMPLEX

    (tw, _), _ = cv2.getTextSize(line1, font, 1.1, 2)
    cv2.putText(frame, line1, (cx - tw // 2, cy - 10), font, 1.1, color, 2, cv2.LINE_AA)

    (tw2, _), _ = cv2.getTextSize(line2, font, 0.6, 2)
    cv2.putText(frame, line2, (cx - tw2 // 2, cy + 30), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# ── Prediction smoothing ──────────────────────────────────────────────────────
WINDOW = 10
prediction_buffer = []

def smooth_prediction(pred):
    prediction_buffer.append(pred)
    if len(prediction_buffer) > WINDOW:
        prediction_buffer.pop(0)
    return max(set(prediction_buffer), key=prediction_buffer.count)

# ══════════════════════════════════════════════════════════════════════════════
#  STATE — rep counting
# ══════════════════════════════════════════════════════════════════════════════
rep_count      = 0
stage          = "up"        # "up" or "down"
rep_angle_min  = float("inf")
rep_angle_max  = float("-inf")
rep_roms       = []          # list of rom% values for current session
last_rom_pct   = None
banner_until   = 0.0         # timestamp until which to show the ROM banner
last_exercise  = None        # detect exercise switches so we can reset

db_conn    = init_db()
session_id = None            # created on first rep

# ── Main loop ─────────────────────────────────────────────────────────────────
cap           = cv2.VideoCapture(0)
prev_time     = 0
current_label  = "Detecting..."
confidence_str = ""
live_angles    = {}

while True:
    success, frame = cap.read()
    if not success:
        break

    frame   = cv2.flip(frame, 1)
    h, w    = frame.shape[:2]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            angles     = extract_angles(results.pose_landmarks, side="left")
            live_angles = angles                          # ← was never assigned before

            X_live = pd.DataFrame([angles])[feature_names]
            pred   = model.predict(X_live)[0]
            proba  = model.predict_proba(X_live).max()

            current_label  = smooth_prediction(pred)
            confidence_str = f"{proba * 100:.0f}%"

        except Exception as e:
            current_label  = "Error"
            confidence_str = str(e)[:50]

        # ── Draw angle values next to every joint ─────────────────────────
        draw_all_joint_angles(frame, results.pose_landmarks, live_angles, w, h, side="left")

        # ── Rep counting + ROM% tracking ──────────────────────────────────
        if current_label in REP_CONFIG and live_angles:

            # Reset counters when exercise changes
            if current_label != last_exercise:
                rep_count     = 0
                stage         = "up"
                rep_angle_min = float("inf")
                rep_angle_max = float("-inf")
                rep_roms      = []
                last_exercise = current_label
                session_id    = None

            cfg   = REP_CONFIG[current_label]
            joint = cfg["joint"]
            angle = live_angles.get(joint)

            if angle is not None:
                # Track range within the current rep attempt
                rep_angle_min = min(rep_angle_min, angle)
                rep_angle_max = max(rep_angle_max, angle)

                # Stage machine  (works for push-ups & squats: angle falls → "down", rises → "up")
                # For pull-ups the sense is the same (elbow angle falls when curled = "down")
                if stage == "up" and angle < cfg["down_thresh"]:
                    stage = "down"

                elif stage == "down" and angle > cfg["up_thresh"]:
                    # ── Rep completed ──────────────────────────────────────
                    stage      = "up"
                    rep_count += 1

                    span        = rep_angle_max - rep_angle_min
                    ideal_span  = cfg["ideal_span"]
                    rom_pct     = min(100.0, (span / ideal_span) * 100.0)
                    last_rom_pct = rom_pct
                    rep_roms.append(rom_pct)
                    banner_until = time.time() + 2.5

                    # Create session on first rep
                    if session_id is None:
                        session_id = save_session_to_db(db_conn, current_label, 0, [])

                    save_rep_to_db(db_conn, session_id, rep_count, current_label,
                                   joint, rep_angle_min, rep_angle_max,
                                   span, ideal_span, rom_pct)

                    # Update session totals
                    avg_rom = sum(rep_roms) / len(rep_roms)
                    db_conn.execute(
                        "UPDATE sessions SET total_reps=?, avg_rom_pct=? WHERE id=?",
                        (rep_count, round(avg_rom, 1), session_id)
                    )
                    db_conn.commit()

                    print(f"[Rep {rep_count}] span={span:.1f}°  ROM={rom_pct:.0f}%")

                    # Reset range for next rep
                    rep_angle_min = float("inf")
                    rep_angle_max = float("-inf")

    # ── ROM bars (live position within ideal window) ──────────────────────────
    if live_angles and current_label in IDEAL_RANGES:
        draw_rom_bars(frame, live_angles, current_label)

    # ── Rep ROM% banner (shown after each rep for 2.5 s) ─────────────────────
    if time.time() < banner_until and last_rom_pct is not None:
        draw_rep_rom_banner(frame, last_rom_pct, rep_count, w, h)

    # ── Rep counter (top-right corner) ────────────────────────────────────────
    if current_label in REP_CONFIG:
        cv2.putText(frame, f"Reps: {rep_count}",
                    (w - 160, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if last_rom_pct is not None:
            color = (0, 220, 100) if last_rom_pct >= 90 else (0, 200, 255) if last_rom_pct >= 70 else (50, 50, 230)
            cv2.putText(frame, f"Last ROM: {last_rom_pct:.0f}%",
                        (w - 200, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # ── FPS ───────────────────────────────────────────────────────────────────
    curr_time = time.time()
    fps       = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # ── Bottom banner ─────────────────────────────────────────────────────────
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
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
db_conn.close()