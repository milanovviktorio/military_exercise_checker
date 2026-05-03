import cv2
import mediapipe as mp
import numpy as np
import joblib
import sqlite3
import time
import pandas as pd
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

IDEAL_RANGES = {
    "Push Ups": {"Elbow_Angle": (100, 180)},
    "Pull ups": {"Elbow_Angle": (60, 180)},
    "Squats":   {"Knee_Angle": (80, 100), "Hip_Angle": (80, 100), "Ankle_Angle": (70, 90)},
}

REP_CONFIG = {
    "Push Ups": {"joint": "Elbow_Angle", "ideal_span": 80},
    "Pull ups": {"joint": "Elbow_Angle", "ideal_span": 120},
    "Squats":   {"joint": "Knee_Angle",  "ideal_span": 80},
}

JOINT_LABELS = {
    "Shoulder_Angle": "Shoulder", "Elbow_Angle": "Elbow",
    "Hip_Angle": "Hip",           "Knee_Angle":  "Knee",
    "Ankle_Angle": "Ankle",       "Shoulder_Ground_Angle": "Shldr(g)",
    "Elbow_Ground_Angle": "Elb(g)", "Hip_Ground_Angle": "Hip(g)",
    "Knee_Ground_Angle": "Knee(g)", "Ankle_Ground_Angle": "Ank(g)",
}

# A rep is counted whenever the tracked joint swings MIN_SWING degrees
# down and then MIN_SWING degrees back up. Lower = more forgiving.
MIN_SWING = 20

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Database ──────────────────────────────────────────────────────────────────

def init_db(path="workout_tracker.db"):
    # isolation_level=None → autocommit: every write is flushed to disk immediately
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            date       TEXT,
            exercise   TEXT,
            total_reps INTEGER,
            avg_rom    REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER,
            rep_number  INTEGER,
            exercise    TEXT,
            rom_percent REAL,
            timestamp   TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    return conn


def db_save_rep(conn, session_id, rep_number, exercise, rom_pct):
    conn.execute(
        "INSERT INTO reps (session_id, rep_number, exercise, rom_percent, timestamp) VALUES (?,?,?,?,?)",
        (session_id, rep_number, exercise, round(rom_pct, 1),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )


def db_upsert_session(conn, session_id, exercise, total_reps, avg_rom):
    if session_id is None:
        cur = conn.execute(
            "INSERT INTO sessions (date, exercise, total_reps, avg_rom) VALUES (?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), exercise, total_reps, round(avg_rom, 1))
        )
        return cur.lastrowid
    conn.execute(
        "UPDATE sessions SET total_reps=?, avg_rom=? WHERE id=?",
        (total_reps, round(avg_rom, 1), session_id)
    )
    return session_id

# ── MediaPipe / model ─────────────────────────────────────────────────────────

model = joblib.load("./Files/model.pkl")
try:
    feature_names = joblib.load("./Files/feature_names.pkl")
except FileNotFoundError:
    feature_names = [
        "Shoulder_Angle", "Elbow_Angle", "Hip_Angle", "Knee_Angle", "Ankle_Angle",
        "Shoulder_Ground_Angle", "Elbow_Ground_Angle", "Hip_Ground_Angle",
        "Knee_Ground_Angle", "Ankle_Ground_Angle",
    ]

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
    d = np.array(b) - np.array(a)
    return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))

def extract_angles(landmarks, side="left"):
    lm = landmarks.landmark
    L  = mp_pose.PoseLandmark
    def xy(lmk): p = lm[lmk.value]; return [p.x, p.y]

    SH, EL, WR = (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST) if side == "left" \
                 else (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST)
    HI, KN, AN, HE = (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE, L.LEFT_HEEL) if side == "left" \
                     else (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE, L.RIGHT_HEEL)

    sh, el, wr = xy(SH), xy(EL), xy(WR)
    hi, kn, an, he = xy(HI), xy(KN), xy(AN), xy(HE)

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

# ── Drawing ───────────────────────────────────────────────────────────────────

def put_text_bg(frame, text, pos, scale=0.45, color=(0, 220, 220), bg=(20, 20, 20)):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x, y = pos
    cv2.rectangle(frame, (x - 3, y - th - 3), (x + tw + 3, y + 3), bg, -1)
    cv2.putText(frame, text, (x, y), FONT, scale, color, 1, cv2.LINE_AA)


def draw_joint_angles(frame, landmarks, angles, w, h, side="left"):
    lm     = landmarks.landmark
    L      = mp_pose.PoseLandmark
    prefix = "LEFT_" if side == "left" else "RIGHT_"

    vertex_map = {
        "Shoulder_Angle":        prefix + "SHOULDER",
        "Elbow_Angle":           prefix + "ELBOW",
        "Hip_Angle":             prefix + "HIP",
        "Knee_Angle":            prefix + "KNEE",
        "Ankle_Angle":           prefix + "ANKLE",
        "Shoulder_Ground_Angle": prefix + "SHOULDER",
        "Elbow_Ground_Angle":    prefix + "ELBOW",
        "Hip_Ground_Angle":      prefix + "HIP",
        "Knee_Ground_Angle":     prefix + "KNEE",
        "Ankle_Ground_Angle":    prefix + "ANKLE",
    }

    y_offset = {}
    for key, angle in angles.items():
        lm_name = vertex_map.get(key)
        if not lm_name:
            continue
        try:
            lm_enum = getattr(L, lm_name)
        except AttributeError:
            continue
        p = lm[lm_enum.value]
        if p.visibility < 0.5:
            continue
        px, py = int(p.x * w), int(p.y * h)
        slot   = (px // 8, py // 8)
        dy     = y_offset.get(slot, 0)
        y_offset[slot] = dy + 16
        put_text_bg(frame, f"{angle:.0f}\u00b0", (px + 10, py - 8 + dy))


def draw_rom_bars(frame, angles, exercise):
    ranges = IDEAL_RANGES.get(exercise, {})
    if not ranges:
        return
    BAR_W, BAR_H, BAR_X, GAP = 170, 14, 10, 28
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (BAR_X * 2 + BAR_W + 130, len(ranges) * GAP + 20), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    y = 20
    for joint, (lo, hi) in ranges.items():
        angle = angles.get(joint, 0.0)
        pct   = max(0.0, min(100.0, (angle - lo) / (hi - lo) * 100)) if hi != lo else 100.0
        ok    = lo <= angle <= hi
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (60, 60, 60), -1)
        if pct:
            cv2.rectangle(frame, (BAR_X, y), (BAR_X + int(BAR_W * pct / 100), y + BAR_H),
                          (0, 200, 80) if ok else (0, 80, 220), -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (120, 120, 120), 1)
        cv2.putText(frame, f"{JOINT_LABELS.get(joint, joint)}: {angle:.0f}\u00b0 ({pct:.0f}%)",
                    (BAR_X + BAR_W + 6, y + BAR_H - 2), FONT, 0.42,
                    (200, 255, 200) if ok else (100, 180, 255), 1)
        y += GAP


def draw_rep_banner(frame, rom_pct, rep_num, w, h):
    if   rom_pct >= 90: label, clr = "FULL ROM",    (0, 220, 100)
    elif rom_pct >= 70: label, clr = "GOOD ROM",    (0, 200, 255)
    elif rom_pct >= 50: label, clr = "PARTIAL ROM", (0, 140, 255)
    else:               label, clr = "POOR ROM",    (50, 50, 230)
    cx, cy = w // 2, h // 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (cx - 175, cy - 65), (cx + 175, cy + 55), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for text, y, scale, color in [
        (f"Rep {rep_num}:  {rom_pct:.0f}% ROM", cy - 10, 1.1, clr),
        (label, cy + 32, 0.6, (255, 255, 255)),
    ]:
        (tw, _), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(frame, text, (cx - tw // 2, y), FONT, scale, color, 2, cv2.LINE_AA)

def draw_confidence_bar(frame, confidence, w, h):
    BAR_W, BAR_H = 200, 12
    x = w - BAR_W - 15
    y = h - 35
    pct  = int(BAR_W * confidence)
    clr  = (0, 220, 100) if confidence >= 0.8 else (0, 200, 255) if confidence >= 0.6 else (50, 50, 230)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + pct,   y + BAR_H), clr, -1)
    cv2.rectangle(frame, (x, y), (x + BAR_W, y + BAR_H), (120, 120, 120), 1)
    cv2.putText(frame, f"Conf: {confidence*100:.0f}%", (x, y - 5), FONT, 0.45, (200, 200, 200), 1)

pred_buffer = []

def smooth_prediction(pred, window=10):
    pred_buffer.append(pred)
    if len(pred_buffer) > window:
        pred_buffer.pop(0)
    return max(set(pred_buffer), key=pred_buffer.count)

# ── Rep state ─────────────────────────────────────────────────────────────────

def make_state():
    return {
        "exercise":      None,
        "phase":         "peak",   # peak → angle is high, waiting to dip
        "peak_angle":    0.0,
        "valley_angle":  float("inf"),
        "count":         0,
        "roms":          [],
        "last_rom":      None,
        "banner_until":  0.0,
        "cooldown_until": 0.0,    # blocks double-counts after a rep completes
        "session_id":    None,
    }

state = make_state()


def on_rep_complete(exercise, span, db_conn):
    cfg     = REP_CONFIG[exercise]
    rom_pct = min(100.0, span / cfg["ideal_span"] * 100.0)

    state["count"]       += 1
    state["last_rom"]     = rom_pct
    state["banner_until"] = time.time() + 2.5
    state["roms"].append(rom_pct)
    avg_rom = sum(state["roms"]) / len(state["roms"])

    state["session_id"] = db_upsert_session(
        db_conn, state["session_id"], exercise, state["count"], avg_rom
    )
    db_save_rep(db_conn, state["session_id"], state["count"], exercise, rom_pct)
    print(f"[Rep {state['count']}]  ROM={rom_pct:.0f}%  (span={span:.1f}°)")


def update_rep_state(angles, exercise, db_conn):
    if exercise != state["exercise"]:
        state.update(make_state())
        state["exercise"] = exercise

    if time.time() < state["cooldown_until"]:
        return

    angle = angles.get(REP_CONFIG[exercise]["joint"])
    if angle is None:
        return

    if state["phase"] == "peak":
        state["peak_angle"] = max(state["peak_angle"], angle)
        if angle < state["peak_angle"] - MIN_SWING:
            state["phase"]        = "valley"
            state["valley_angle"] = angle

    elif state["phase"] == "valley":
        state["valley_angle"] = min(state["valley_angle"], angle)
        if angle > state["valley_angle"] + MIN_SWING:
            span = state["peak_angle"] - state["valley_angle"]
            on_rep_complete(exercise, span, db_conn)
            state["phase"]         = "peak"
            state["peak_angle"]    = 0.0      # reset so it must earn a new peak
            state["cooldown_until"] = time.time() + 0.8

# ── Main loop ─────────────────────────────────────────────────────────────────

db_conn       = init_db()
cap           = cv2.VideoCapture(0)
prev_time     = 0.0
current_label = "Detecting..."
conf_val      = 0.0
conf_str      = ""
live_angles   = {}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame  = cv2.flip(frame, 1)
    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = pose.process(rgb)
    rgb.flags.writeable = True

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            live_angles   = extract_angles(results.pose_landmarks, side="left")
            X             = pd.DataFrame([live_angles])[feature_names]
            current_label = smooth_prediction(model.predict(X)[0])
            conf_val      = float(model.predict_proba(X).max())
            conf_str      = f"{conf_val * 100:.0f}%"
        except Exception as e:
            current_label = "Error"
            conf_str      = str(e)[:50]

        draw_joint_angles(frame, results.pose_landmarks, live_angles, w, h, side="left")

        if current_label in REP_CONFIG and live_angles:
            update_rep_state(live_angles, current_label, db_conn)

    if live_angles and current_label in IDEAL_RANGES:
        draw_rom_bars(frame, live_angles, current_label)

    if time.time() < state["banner_until"] and state["last_rom"] is not None:
        draw_rep_banner(frame, state["last_rom"], state["count"], w, h)

    if current_label in REP_CONFIG:
        cv2.putText(frame, f"Reps: {state['count']}",
                    (w - 155, 75), FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if state["last_rom"] is not None:
            r   = state["last_rom"]
            clr = (0, 220, 100) if r >= 90 else (0, 200, 255) if r >= 70 else (50, 50, 230)
            cv2.putText(frame, f"Last ROM: {r:.0f}%",
                        (w - 195, 105), FONT, 0.7, clr, 2, cv2.LINE_AA)

    now       = time.time()
    fps       = 1 / (now - prev_time) if prev_time else 0
    prev_time = now

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, f"Exercise: {current_label}", (15, h - 55), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 180), 2)
    cv2.putText(frame, f"FPS: {int(fps)}",           (w - 110, 35), FONT, 0.8, (0, 255, 0), 2)
    draw_confidence_bar(frame, conf_val, w, h)

    cv2.imshow("Exercise Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
db_conn.close()