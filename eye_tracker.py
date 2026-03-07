import cv2
import numpy as np
import time
import os
import json
import threading
import urllib.request
from collections import deque
from typing import Optional, Tuple

from flask import Flask, request, Response, jsonify

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# ── Config ──────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PORT = 5000
BUFFER_SIZE = 1000

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_model.json")

# MediaPipe landmark indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

# Kalman tuning
Q_POS = 0.005
R_MEASURE = 0.05


# ── Model download ─────────────────────────────────────────────────────────
def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")


# ── IrisTracker (MediaPipe) ────────────────────────────────────────────────
class IrisTracker:
    def __init__(self):
        _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        h, w = frame.shape[:2]
        lx, ly = self._iris_ratio(landmarks, LEFT_IRIS, LEFT_EYE_CORNERS, w, h)
        rx, ry = self._iris_ratio(landmarks, RIGHT_IRIS, RIGHT_EYE_CORNERS, w, h)
        return (lx + rx) / 2.0, (ly + ry) / 2.0

    def _iris_ratio(self, landmarks, iris_ids, corner_ids, w, h):
        iris_center = np.mean(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in iris_ids], axis=0
        )
        corner_inner = np.array(
            [landmarks[corner_ids[0]].x * w, landmarks[corner_ids[0]].y * h]
        )
        corner_outer = np.array(
            [landmarks[corner_ids[1]].x * w, landmarks[corner_ids[1]].y * h]
        )
        eye_width = np.linalg.norm(corner_outer - corner_inner)
        if eye_width < 1:
            return 0.5, 0.5
        eye_vec = corner_outer - corner_inner
        iris_vec = iris_center - corner_inner
        ratio_x = np.dot(iris_vec, eye_vec) / (eye_width**2)
        eye_normal = np.array([-eye_vec[1], eye_vec[0]]) / eye_width
        ratio_y = np.dot(iris_vec, eye_normal) / eye_width
        return ratio_x, ratio_y


# ── Kalman Filter (adapted from imu_kalman.py) ────────────────────────────
class GazeKalman:
    """Simple 1-state Kalman filter per axis for gaze smoothing."""

    def __init__(self, q=Q_POS, r=R_MEASURE):
        self.q = q
        self.r = r
        self.x_est = None
        self.p_x = 1.0
        self.y_est = None
        self.p_y = 1.0

    def update(self, mx: float, my: float) -> Tuple[float, float]:
        if self.x_est is None:
            self.x_est = mx
            self.y_est = my
            return mx, my

        # Predict (state unchanged, covariance grows)
        self.p_x += self.q
        self.p_y += self.q

        # Update X
        k_x = self.p_x / (self.p_x + self.r)
        self.x_est += k_x * (mx - self.x_est)
        self.p_x *= 1.0 - k_x

        # Update Y
        k_y = self.p_y / (self.p_y + self.r)
        self.y_est += k_y * (my - self.y_est)
        self.p_y *= 1.0 - k_y

        return self.x_est, self.y_est

    def reset(self):
        self.x_est = None
        self.y_est = None
        self.p_x = 1.0
        self.p_y = 1.0


# ── Calibration Model ─────────────────────────────────────────────────────
class CalibrationModel:
    def __init__(self):
        self.samples_iris = []
        self.samples_screen = []
        self.poly = PolynomialFeatures(degree=2, include_bias=True)
        self.model_x = Ridge(alpha=1.0)
        self.model_y = Ridge(alpha=1.0)
        self.is_trained = False

    def add_sample(self, iris_x, iris_y, screen_x, screen_y):
        self.samples_iris.append([iris_x, iris_y])
        self.samples_screen.append([screen_x, screen_y])

    def train(self):
        X = self.poly.fit_transform(np.array(self.samples_iris))
        Y = np.array(self.samples_screen)
        self.model_x.fit(X, Y[:, 0])
        self.model_y.fit(X, Y[:, 1])
        self.is_trained = True

    def predict(self, iris_x, iris_y) -> Tuple[float, float]:
        X = self.poly.transform(np.array([[iris_x, iris_y]]))
        sx = float(self.model_x.predict(X)[0])
        sy = float(self.model_y.predict(X)[0])
        return sx, sy

    def clear(self):
        self.samples_iris.clear()
        self.samples_screen.clear()
        self.is_trained = False

    def save(self, path):
        data = {
            "samples_iris": self.samples_iris,
            "samples_screen": self.samples_screen,
            "coef_x": self.model_x.coef_.tolist(),
            "intercept_x": float(self.model_x.intercept_),
            "coef_y": self.model_y.coef_.tolist(),
            "intercept_y": float(self.model_y.intercept_),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.samples_iris = data["samples_iris"]
        self.samples_screen = data["samples_screen"]
        # Retrain from saved samples to restore poly + models properly
        if len(self.samples_iris) >= 9:
            self.train()
            return True
        return False


# ── Shared State ───────────────────────────────────────────────────────────
gaze_lock = threading.Lock()
latest_gaze = {"x": 0.5, "y": 0.5, "tracking": False, "ts": 0}
gaze_buffer = deque(maxlen=BUFFER_SIZE)

iris_lock = threading.Lock()
iris_buffer = deque(maxlen=10)

frame_lock = threading.Lock()
latest_frame = None  # JPEG bytes

calibration = CalibrationModel()
kalman = GazeKalman()


# ── Camera Thread ──────────────────────────────────────────────────────────
def camera_thread():
    global latest_gaze, latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Could not open camera at index", CAMERA_INDEX)
        return
    print(
        f"[OK] Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )

    tracker = IrisTracker()
    print("[OK] Face landmarker model loaded")

    face_lost_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)

        # Encode JPEG for MJPEG stream
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with frame_lock:
            latest_frame = jpeg.tobytes()

        # Iris detection
        result = tracker.process(frame)

        if result is None:
            face_lost_frames += 1
            if face_lost_frames > 30:
                with gaze_lock:
                    latest_gaze = {**latest_gaze, "tracking": False}
            continue

        face_lost_frames = 0
        iris_x, iris_y = result

        # Store iris data for calibration
        with iris_lock:
            iris_buffer.append((iris_x, iris_y, time.time()))

        # If calibrated, predict screen coordinates
        if calibration.is_trained:
            sx, sy = calibration.predict(iris_x, iris_y)
            sx = max(0.0, min(1.0, sx))
            sy = max(0.0, min(1.0, sy))
            sx, sy = kalman.update(sx, sy)

            entry = {
                "x": round(float(sx), 5),
                "y": round(float(sy), 5),
                "tracking": True,
                "ts": round(time.time() * 1000),
            }
            with gaze_lock:
                latest_gaze = entry
                gaze_buffer.append(entry)


# ── MJPEG Stream ───────────────────────────────────────────────────────────
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.03)
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)


# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<title>GameSense Eye Tracker</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #ccc;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden; width: 100vw; height: 100vh;
  }
  .screen { display: none; width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
  .screen.active { display: flex; flex-direction: column; align-items: center; justify-content: center; }

  /* Setup */
  #setup-screen .cam-preview {
    border-radius: 12px; border: 2px solid #333; margin-bottom: 20px;
  }
  #setup-screen .instructions { text-align: center; max-width: 500px; }
  #setup-screen h1 { color: #e0e0e0; font-size: 28px; margin-bottom: 12px; }
  #setup-screen p { color: #999; font-size: 15px; line-height: 1.5; margin-bottom: 10px; }
  .face-status {
    font-size: 18px; font-weight: 600; margin: 16px 0;
    padding: 10px 24px; border-radius: 8px; display: inline-block;
  }
  .face-ok { background: #00ff8822; color: #00ff88; border: 1px solid #00ff8844; }
  .face-bad { background: #ff444422; color: #ff4444; border: 1px solid #ff444444; }
  .btn-primary {
    padding: 14px 40px; font-size: 18px; background: #00cc66; color: #111;
    border: none; border-radius: 8px; cursor: pointer; font-weight: 600;
    transition: all 0.2s; margin: 6px;
  }
  .btn-primary:disabled { opacity: 0.3; cursor: not-allowed; }
  .btn-primary:not(:disabled):hover { background: #00ff88; transform: scale(1.05); }
  .btn-secondary {
    padding: 10px 30px; font-size: 14px; background: transparent; color: #888;
    border: 1px solid #444; border-radius: 8px; cursor: pointer;
    transition: all 0.2s; margin: 6px;
  }
  .btn-secondary:hover { border-color: #888; color: #ccc; }

  /* Calibration */
  #calibration-screen { cursor: crosshair; }
  .cal-header {
    position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
    text-align: center; z-index: 100; pointer-events: none;
  }
  .cal-header h2 { color: #e0e0e0; font-size: 22px; }
  .cal-header p { color: #888; font-size: 14px; margin-top: 6px; }
  .cal-point {
    position: absolute; width: 40px; height: 40px; border-radius: 50%;
    transform: translate(-50%, -50%); cursor: pointer; z-index: 50;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #fff; transition: all 0.3s;
  }
  .cal-point.waiting { background: #ffffff22; border: 2px solid #ffffff44; }
  .cal-point.active {
    background: #ffaa00; border: 2px solid #ffcc44;
    box-shadow: 0 0 20px #ffaa0066, 0 0 40px #ffaa0033;
    animation: pulse 1s ease-in-out infinite;
  }
  .cal-point.done { background: #00cc66; border: 2px solid #00ff88; box-shadow: 0 0 10px #00ff8844; }
  @keyframes pulse {
    0%, 100% { transform: translate(-50%, -50%) scale(1); }
    50% { transform: translate(-50%, -50%) scale(1.3); }
  }
  #cal-progress {
    position: fixed; bottom: 30px; left: 50%; transform: translateX(-50%);
    font-size: 16px; color: #888; z-index: 100; pointer-events: none;
  }

  /* Validation */
  .val-point {
    position: absolute; width: 20px; height: 20px; border-radius: 50%;
    background: #00aaff; border: 2px solid #44ccff;
    box-shadow: 0 0 15px #00aaff66;
    transform: translate(-50%, -50%); display: none; z-index: 50;
  }
  #accuracy-result { font-size: 48px; font-weight: 700; margin: 20px 0; }
  .accuracy-excellent { color: #00ff88; }
  .accuracy-good { color: #ffaa00; }
  .accuracy-poor { color: #ff4444; }

  /* Tracking */
  #tracking-screen { cursor: default; }
  #gaze-dot {
    position: absolute; width: 30px; height: 30px;
    background: radial-gradient(circle, #00ff88, #00cc66);
    border-radius: 50%; transform: translate(-50%, -50%);
    box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff8844;
    pointer-events: none; z-index: 100;
    transition: left 0.08s linear, top 0.08s linear;
  }
  #trail-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  #crosshair-h, #crosshair-v { position: absolute; background: #ffffff11; }
  #crosshair-h { width: 100%; height: 1px; top: 50%; }
  #crosshair-v { height: 100%; width: 1px; left: 50%; }
  #tracking-status {
    position: fixed; top: 20px; left: 20px; color: #888;
    font-family: monospace; font-size: 14px; z-index: 200; pointer-events: none;
  }
  .top-right-btn {
    position: fixed; top: 20px; right: 20px; z-index: 200;
    padding: 8px 20px; font-size: 13px; background: #333; color: #aaa;
    border: 1px solid #555; border-radius: 6px; cursor: pointer;
  }
  .top-right-btn:hover { background: #444; color: #fff; }
  #bg-message {
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: #00ff8822; color: #00ff88; padding: 10px 24px; border-radius: 8px;
    font-size: 14px; z-index: 200; pointer-events: none; border: 1px solid #00ff8844;
  }
</style>
</head>
<body>

<!-- Screen 1: Setup -->
<div id="setup-screen" class="screen active">
  <img id="cam-preview" class="cam-preview" src="/video_feed" width="480" height="360" alt="Camera">
  <div class="instructions">
    <h1>GameSense Eye Tracker</h1>
    <p>Position your face in the camera frame above.<br>
       Keep a comfortable distance from the screen.</p>
    <div id="face-status" class="face-status face-bad">Waiting for face...</div>
    <div>
      <button id="btn-begin-cal" class="btn-primary" disabled>Begin Calibration</button>
    </div>
    <div>
      <button id="btn-use-prev" class="btn-secondary" style="display:none">Use Previous Calibration</button>
    </div>
  </div>
</div>

<!-- Screen 2: Calibration -->
<div id="calibration-screen" class="screen">
  <div class="cal-header">
    <h2>Calibration</h2>
    <p>Look at the orange dot and click it 5 times</p>
  </div>
  <div id="cal-points-container"></div>
  <div id="cal-progress">0 / 9 points complete</div>
</div>

<!-- Screen 3: Validation -->
<div id="validation-screen" class="screen">
  <div class="val-point" id="val-point"></div>
  <div id="val-results" style="text-align:center; display:none;">
    <h2 style="color:#e0e0e0; font-size:24px;">Calibration Quality</h2>
    <div id="accuracy-result"></div>
    <p id="accuracy-desc" style="color:#999; font-size:16px;"></p>
    <div style="margin-top:20px;">
      <button class="btn-secondary" id="btn-recal-val">Recalibrate</button>
      <button class="btn-primary" id="btn-start-tracking">Start Tracking</button>
    </div>
  </div>
  <div id="val-status" style="text-align:center; display:none;">
    <h2 style="color:#e0e0e0;">Validating Accuracy...</h2>
    <p style="color:#888; margin-top:8px;">Look at the blue dot</p>
    <p style="color:#666; margin-top:4px;" id="val-progress-text">Point 1 / 5</p>
  </div>
</div>

<!-- Screen 4: Tracking -->
<div id="tracking-screen" class="screen">
  <div id="crosshair-h"></div>
  <div id="crosshair-v"></div>
  <canvas id="trail-canvas"></canvas>
  <div id="gaze-dot"></div>
  <div id="tracking-status">Waiting for gaze...</div>
  <button class="top-right-btn" id="btn-recal-tracking">Recalibrate</button>
  <div id="bg-message">Tracking runs in background. You can minimize or close this tab.</div>
</div>

<script>
(function() {
  let currentScreen = 'setup';

  function showScreen(name) {
    currentScreen = name;
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(name + '-screen').classList.add('active');
  }

  // ── Setup screen ──
  let faceCheckInterval = setInterval(() => {
    fetch('/status').then(r => r.json()).then(d => {
      const el = document.getElementById('face-status');
      const btn = document.getElementById('btn-begin-cal');
      if (d.face_detected) {
        el.textContent = 'Face detected - ready!';
        el.className = 'face-status face-ok';
        btn.disabled = false;
      } else {
        el.textContent = 'No face detected - adjust position';
        el.className = 'face-status face-bad';
        if (currentScreen === 'setup') btn.disabled = true;
      }
      if (d.calibrated) {
        document.getElementById('btn-use-prev').style.display = 'inline-block';
      }
    }).catch(() => {});
  }, 500);

  // ── Calibration ──
  const CAL_POSITIONS = [
    [10,10],[50,10],[90,10],
    [10,50],[50,50],[90,50],
    [10,90],[50,90],[90,90]
  ];
  const CLICKS_PER_POINT = 5;
  let calOrder = [], calIndex = 0, calClicks = {};

  function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function startCalibration() {
    // Reset server calibration state
    fetch('/calibrate_reset', { method: 'POST' });
    showScreen('calibration');

    calOrder = shuffle([0,1,2,3,4,5,6,7,8]);
    calIndex = 0;
    calClicks = {};
    for (let i = 0; i < 9; i++) calClicks[i] = 0;

    const container = document.getElementById('cal-points-container');
    container.innerHTML = '';

    CAL_POSITIONS.forEach((pos, i) => {
      const pt = document.createElement('div');
      pt.className = 'cal-point waiting';
      pt.id = 'cal-pt-' + i;
      pt.style.left = pos[0] + 'vw';
      pt.style.top = pos[1] + 'vh';
      pt.textContent = '0/' + CLICKS_PER_POINT;
      pt.addEventListener('click', () => onCalClick(i));
      container.appendChild(pt);
    });
    highlightCalPoint();
  }

  function highlightCalPoint() {
    document.querySelectorAll('.cal-point').forEach(p => {
      if (!p.classList.contains('done')) p.className = 'cal-point waiting';
    });
    if (calIndex < 9) {
      document.getElementById('cal-pt-' + calOrder[calIndex]).className = 'cal-point active';
    }
    const done = Object.values(calClicks).filter(c => c >= CLICKS_PER_POINT).length;
    document.getElementById('cal-progress').textContent = done + ' / 9 points complete';
  }

  function onCalClick(idx) {
    if (calOrder[calIndex] !== idx) return;

    const pos = CAL_POSITIONS[idx];
    // Send calibration point to server
    fetch('/calibrate_point', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ screen_x: pos[0] / 100, screen_y: pos[1] / 100 })
    });

    calClicks[idx]++;
    const pt = document.getElementById('cal-pt-' + idx);
    pt.textContent = calClicks[idx] + '/' + CLICKS_PER_POINT;

    if (calClicks[idx] >= CLICKS_PER_POINT) {
      pt.className = 'cal-point done';
      pt.textContent = '\u2713';
      calIndex++;
      if (calIndex >= 9) {
        finishCalibration();
        return;
      }
    }
    highlightCalPoint();
  }

  function finishCalibration() {
    document.getElementById('cal-progress').textContent = 'Training model...';
    fetch('/calibrate_finish', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        if (d.ok) startValidation();
        else alert('Calibration error: ' + (d.error || 'unknown'));
      });
  }

  // ── Validation ──
  const VAL_POSITIONS = [[30,30],[70,30],[50,50],[30,70],[70,70]];
  let valIndex = 0, valErrors = [];

  function startValidation() {
    showScreen('validation');
    document.getElementById('val-results').style.display = 'none';
    document.getElementById('val-status').style.display = 'block';
    valIndex = 0;
    valErrors = [];
    setTimeout(showValPoint, 500);
  }

  function showValPoint() {
    if (valIndex >= VAL_POSITIONS.length) { finishValidation(); return; }

    document.getElementById('val-progress-text').textContent =
      'Point ' + (valIndex + 1) + ' / ' + VAL_POSITIONS.length;

    const pos = VAL_POSITIONS[valIndex];
    const pt = document.getElementById('val-point');
    pt.style.left = pos[0] + 'vw';
    pt.style.top = pos[1] + 'vh';
    pt.style.display = 'block';

    const targetX = pos[0] / 100;
    const targetY = pos[1] / 100;
    let samples = [];
    let collectCount = 0;

    const interval = setInterval(() => {
      fetch('/gaze').then(r => r.json()).then(d => {
        if (d.tracking) {
          // Normalize pixel coords to 0-1 for comparison
          samples.push({ x: d.x, y: d.y });
        }
        collectCount++;
      }).catch(() => {});
    }, 50);

    setTimeout(() => {
      clearInterval(interval);
      pt.style.display = 'none';

      if (samples.length > 3) {
        const avgX = samples.reduce((s, p) => s + p.x, 0) / samples.length;
        const avgY = samples.reduce((s, p) => s + p.y, 0) / samples.length;
        const error = Math.sqrt((avgX - targetX) ** 2 + (avgY - targetY) ** 2);
        valErrors.push(error);
      }

      valIndex++;
      setTimeout(showValPoint, 300);
    }, 2000);
  }

  function finishValidation() {
    document.getElementById('val-status').style.display = 'none';
    document.getElementById('val-results').style.display = 'block';

    const avgError = valErrors.length > 0
      ? valErrors.reduce((a, b) => a + b, 0) / valErrors.length : 1;

    // Error is in normalized coords (0-1). Convert to accuracy %.
    // ~0.05 error = excellent, ~0.15 = poor
    const accuracy = Math.max(0, Math.round(100 - avgError * 500));

    const resEl = document.getElementById('accuracy-result');
    const descEl = document.getElementById('accuracy-desc');
    resEl.textContent = accuracy + '%';

    if (accuracy >= 75) {
      resEl.className = 'accuracy-excellent';
      descEl.textContent = 'Excellent! Ready for tracking.';
    } else if (accuracy >= 45) {
      resEl.className = 'accuracy-good';
      descEl.textContent = 'Good accuracy. Recalibration may improve results.';
    } else {
      resEl.className = 'accuracy-poor';
      descEl.textContent = 'Low accuracy. Try recalibrating with better lighting.';
    }
  }

  // ── Tracking ──
  let trailCtx = null, prevX = null, prevY = null;

  function startTracking() {
    showScreen('tracking');
    const canvas = document.getElementById('trail-canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    trailCtx = canvas.getContext('2d');
    prevX = null; prevY = null;

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });

    pollGaze();
  }

  function pollGaze() {
    if (currentScreen !== 'tracking') return;

    fetch('/gaze').then(r => r.json()).then(d => {
      const dot = document.getElementById('gaze-dot');
      const status = document.getElementById('tracking-status');

      if (d.tracking) {
        const px = d.x * window.innerWidth;
        const py = d.y * window.innerHeight;
        dot.style.left = px + 'px';
        dot.style.top = py + 'px';
        dot.style.opacity = '1';
        status.textContent = 'x: ' + d.x.toFixed(4) + '  y: ' + d.y.toFixed(4);
        status.style.color = '#00ff88';

        if (trailCtx && prevX !== null) {
          trailCtx.strokeStyle = 'rgba(0,255,136,0.15)';
          trailCtx.lineWidth = 2;
          trailCtx.beginPath();
          trailCtx.moveTo(prevX, prevY);
          trailCtx.lineTo(px, py);
          trailCtx.stroke();
        }
        if (trailCtx) {
          trailCtx.fillStyle = 'rgba(26,26,46,0.02)';
          trailCtx.fillRect(0, 0, trailCtx.canvas.width, trailCtx.canvas.height);
        }
        prevX = px; prevY = py;
      } else {
        status.textContent = 'Face not detected';
        status.style.color = '#ff4444';
        dot.style.opacity = '0.3';
        prevX = null; prevY = null;
      }

      requestAnimationFrame(pollGaze);
    }).catch(() => setTimeout(pollGaze, 100));
  }

  // ── Button wiring ──
  document.getElementById('btn-begin-cal').addEventListener('click', startCalibration);
  document.getElementById('btn-use-prev').addEventListener('click', startTracking);
  document.getElementById('btn-recal-val').addEventListener('click', startCalibration);
  document.getElementById('btn-start-tracking').addEventListener('click', startTracking);
  document.getElementById('btn-recal-tracking').addEventListener('click', startCalibration);
})();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    with iris_lock:
        recent = list(iris_buffer)
    face_detected = len(recent) > 0 and (time.time() - recent[-1][2]) < 1.0 if recent else False
    return jsonify({
        "face_detected": face_detected,
        "calibrated": calibration.is_trained,
        "sample_count": len(calibration.samples_iris),
    })


@app.route("/calibrate_point", methods=["POST"])
def calibrate_point():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "no data"}), 400

    screen_x = float(data["screen_x"])
    screen_y = float(data["screen_y"])

    # Average recent iris readings for stability
    with iris_lock:
        recent = list(iris_buffer)[-5:]
    if not recent:
        return jsonify({"error": "no iris data - face not detected"}), 400

    avg_ix = sum(r[0] for r in recent) / len(recent)
    avg_iy = sum(r[1] for r in recent) / len(recent)

    calibration.add_sample(avg_ix, avg_iy, screen_x, screen_y)
    return jsonify({"ok": True, "samples": len(calibration.samples_iris)})


@app.route("/calibrate_finish", methods=["POST"])
def calibrate_finish():
    if len(calibration.samples_iris) < 9:
        return jsonify({"ok": False, "error": "not enough samples"}), 400
    calibration.train()
    calibration.save(CALIBRATION_PATH)
    kalman.reset()
    print(f"[CALIBRATION] Trained with {len(calibration.samples_iris)} samples, saved to {CALIBRATION_PATH}")
    return jsonify({"ok": True})


@app.route("/calibrate_reset", methods=["POST"])
def calibrate_reset():
    calibration.clear()
    kalman.reset()
    return jsonify({"ok": True})


@app.route("/gaze")
def get_gaze():
    with gaze_lock:
        return Response(json.dumps(latest_gaze), mimetype="application/json")


@app.route("/gaze_history")
def gaze_history():
    n = request.args.get("n", 100, type=int)
    with gaze_lock:
        items = list(gaze_buffer)[-n:]
    return Response(json.dumps(items), mimetype="application/json")


if __name__ == "__main__":
    # Load previous calibration if exists
    if os.path.exists(CALIBRATION_PATH):
        if calibration.load(CALIBRATION_PATH):
            print(f"[OK] Loaded previous calibration ({len(calibration.samples_iris)} samples)")
        else:
            print("[WARN] Previous calibration file found but insufficient data")

    # Start camera thread
    threading.Thread(target=camera_thread, daemon=True).start()

    print()
    print("  Eye Tracker running on http://localhost:5000")
    print("  Open browser to calibrate. After calibration, you can close the browser.")
    print("  GET /gaze returns live eye tracking coordinates.")
    print()

    app.run(host="0.0.0.0", port=PORT, debug=False)
