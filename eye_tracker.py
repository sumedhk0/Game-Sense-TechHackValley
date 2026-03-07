import cv2
import numpy as np
import time
import os
import json
import threading
import urllib.request
from typing import Optional, Tuple
from flask import Flask, Response

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat

#config
CAMERA_INDEX=0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

DEADZONE_RADIUS = 0.07
SENSITIVITY=800.0
MAX_SPEED=1500.0
ACCELERATION_CURVE=1.5

GAZE_SMOOTHING=0.35
OUTPUT_SMOOTHING=0.5

CENTER_X=0.5
CENTER_Y=0.5

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

#MEDIAPIPE landmark indices
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

#eye tracker
def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

class IrisTracker:
    def __init__(self):
        _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self,frame: np.ndarray) ->Optional[Tuple[float,float]]:
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.landmarker.detect_for_video(mp_image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        landmarks = result.face_landmarks[0]
        h,w = frame.shape[:2]
        lx,ly = self._iris_ratio(landmarks,LEFT_IRIS,LEFT_EYE_CORNERS,w,h)
        rx,ry = self._iris_ratio(landmarks,RIGHT_IRIS, RIGHT_EYE_CORNERS,w,h)
        return (lx+rx)/2.0,(ly+ry)/2.0

    def _iris_ratio(self,landmarks,iris_ids, corner_ids, w,h):
        iris_center=np.mean([[landmarks[i].x * w,landmarks[i].y*h] for i in iris_ids], axis=0)
        corner_inner = np.array([landmarks[corner_ids[0]].x * w,landmarks[corner_ids[0]].y*h])
        corner_outer = np.array([landmarks[corner_ids[1]].x * w,landmarks[corner_ids[1]].y*h])
        eye_width = np.linalg.norm(corner_outer-corner_inner)
        if eye_width < 1:
            return 0.5,0.5

        eye_vec = corner_outer-corner_inner
        iris_vec=iris_center-corner_inner
        ratio_x=np.dot(iris_vec,eye_vec)/(eye_width**2)
        eye_normal = np.array([-eye_vec[1],eye_vec[0]])/eye_width
        ratio_y=np.dot(iris_vec,eye_normal)/eye_width
        return ratio_x,ratio_y
    
#smoothing
class EMA:
    def __init__(self,alpha: float):
        self.alpha=alpha
        self.value: Optional[Tuple[float,float]]=None
    
    def update(self,x: float,y: float)-> Tuple[float,float]:
        if self.value is None:
            self.value= (x,y)
        else:
            self.value = (
                self.alpha*x + (1-self.alpha) * self.value[0],
                self.alpha*y + (1-self.alpha) * self.value[1],
            )
        return self.value
    
    def reset(self):
        self.value=None
        

latest_gaze = {"x": 0.5, "y": 0.5, "tracking": False}

app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head><title>Eye Tracker</title>
<style>
  * { margin: 0; padding: 0; }
  body { background: #1a1a2e; overflow: hidden; }
  #dot {
    position: absolute; width: 30px; height: 30px;
    background: radial-gradient(circle, #00ff88, #00cc66);
    border-radius: 50%; transform: translate(-50%, -50%);
    box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff8844;
    transition: left 0.05s linear, top 0.05s linear;
  }
  #trail { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  #status {
    position: fixed; top: 20px; left: 20px; color: #888;
    font-family: monospace; font-size: 14px;
  }
  #crosshair-h, #crosshair-v {
    position: absolute; background: #ffffff11;
  }
  #crosshair-h { width: 100%; height: 1px; top: 50%; }
  #crosshair-v { height: 100%; width: 1px; left: 50%; }
</style>
</head>
<body>
  <div id="crosshair-h"></div>
  <div id="crosshair-v"></div>
  <canvas id="trail"></canvas>
  <div id="dot"></div>
  <div id="status">Waiting for face...</div>
  <script>
    const dot = document.getElementById('dot');
    const status = document.getElementById('status');
    const canvas = document.getElementById('trail');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    window.onresize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; };

    let prevX = null, prevY = null;

    function poll() {
      fetch('/gaze').then(r => r.json()).then(d => {
        const px = d.x * window.innerWidth;
        const py = d.y * window.innerHeight;
        dot.style.left = px + 'px';
        dot.style.top = py + 'px';
        if (d.tracking) {
          status.textContent = `x: ${d.x.toFixed(4)}  y: ${d.y.toFixed(4)}`;
          status.style.color = '#00ff88';
          dot.style.opacity = '1';
          if (prevX !== null) {
            ctx.strokeStyle = 'rgba(0,255,136,0.15)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(px, py);
            ctx.stroke();
          }
          prevX = px; prevY = py;
          // fade trail
          ctx.fillStyle = 'rgba(26,26,46,0.02)';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        } else {
          status.textContent = 'No face detected';
          status.style.color = '#ff4444';
          dot.style.opacity = '0.3';
          prevX = null; prevY = null;
        }
        requestAnimationFrame(poll);
      }).catch(() => setTimeout(poll, 100));
    }
    poll();
  </script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/gaze')
def gaze():
    return Response(json.dumps(latest_gaze), mimetype='application/json')

def tracker_loop():
    global latest_gaze
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,60)

    if not cap.isOpened():
        print("[ERROR] Could not open camera at index", CAMERA_INDEX)
        return
    print(f"[OK] Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    tracker=IrisTracker()
    print("[OK] Face landmarker model loaded")
    gaze_filter = EMA(alpha=GAZE_SMOOTHING)
    face_lost_frames = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            continue
        frame = cv2.flip(frame,1)
        frame_count += 1
        result = tracker.process(frame)

        if result is None:
            face_lost_frames+=1
            if face_lost_frames == 1 or face_lost_frames % 60 == 0:
                print(f"[DEBUG] No face detected (lost for {face_lost_frames} frames, total frames: {frame_count})")
            if face_lost_frames > 30:
                gaze_filter.reset()
                latest_gaze = {"x": latest_gaze["x"], "y": latest_gaze["y"], "tracking": False}
            continue
        if face_lost_frames > 0:
            print(f"[OK] Face detected again after {face_lost_frames} frames")
        face_lost_frames=0
        iris_x,iris_y=result

        smooth_x, smooth_y = gaze_filter.update(iris_x,iris_y)
        latest_gaze = {"x": float(smooth_x), "y": float(smooth_y), "tracking": True}

if __name__ == "__main__":
    threading.Thread(target=tracker_loop, daemon=True).start()
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=5000, debug=False)