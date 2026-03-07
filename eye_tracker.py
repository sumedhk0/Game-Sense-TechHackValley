import json
import time
import threading
from collections import deque
from flask import Flask, request, Response, jsonify

# Config
BUFFER_SIZE = 1000
PORT = 5000

# Shared state
gaze_lock = threading.Lock()
latest_gaze = {"x": 0, "y": 0, "tracking": False, "ts": 0}
gaze_buffer = deque(maxlen=BUFFER_SIZE)

app = Flask(__name__)

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
<title>GameSense Eye Tracker</title>
<script src="https://cdn.jsdelivr.net/npm/webgazer@2.1.3/dist/webgazer.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #ccc;
    font-family: 'Segoe UI', system-ui, sans-serif;
    overflow: hidden; width: 100vw; height: 100vh;
  }
  .screen { display: none; width: 100%; height: 100%; position: absolute; top: 0; left: 0; }
  .screen.active { display: flex; flex-direction: column; align-items: center; justify-content: center; }

  /* Screen 1: Setup */
  #setup-screen .instructions {
    text-align: center; margin-top: 20px; max-width: 500px;
  }
  #setup-screen .instructions h1 { color: #e0e0e0; font-size: 28px; margin-bottom: 16px; }
  #setup-screen .instructions p { color: #999; font-size: 16px; line-height: 1.6; margin-bottom: 12px; }
  #setup-screen .face-status {
    font-size: 18px; font-weight: 600; margin: 20px 0;
    padding: 10px 24px; border-radius: 8px;
  }
  .face-ok { background: #00ff8822; color: #00ff88; border: 1px solid #00ff8844; }
  .face-bad { background: #ff444422; color: #ff4444; border: 1px solid #ff444444; }
  #begin-calibration {
    margin-top: 20px; padding: 14px 40px; font-size: 18px;
    background: #00cc66; color: #111; border: none; border-radius: 8px;
    cursor: pointer; font-weight: 600; transition: all 0.2s;
  }
  #begin-calibration:disabled { opacity: 0.3; cursor: not-allowed; }
  #begin-calibration:not(:disabled):hover { background: #00ff88; transform: scale(1.05); }
  #use-previous {
    margin-top: 12px; padding: 10px 30px; font-size: 14px;
    background: transparent; color: #888; border: 1px solid #444; border-radius: 8px;
    cursor: pointer; transition: all 0.2s; display: none;
  }
  #use-previous:hover { border-color: #888; color: #ccc; }

  /* Screen 2: Calibration */
  #calibration-screen { cursor: crosshair; }
  #calibration-screen .cal-header {
    position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
    text-align: center; z-index: 100; pointer-events: none;
  }
  #calibration-screen .cal-header h2 { color: #e0e0e0; font-size: 22px; }
  #calibration-screen .cal-header p { color: #888; font-size: 14px; margin-top: 6px; }
  .cal-point {
    position: absolute; width: 40px; height: 40px; border-radius: 50%;
    transform: translate(-50%, -50%); cursor: pointer; z-index: 50;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 700; color: #fff;
    transition: all 0.3s;
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

  /* Screen 3: Validation */
  #validation-screen .val-content { text-align: center; }
  #validation-screen h2 { color: #e0e0e0; font-size: 24px; margin-bottom: 16px; }
  #validation-screen p { color: #999; font-size: 16px; margin-bottom: 8px; }
  .val-point {
    position: absolute; width: 20px; height: 20px; border-radius: 50%;
    background: #00aaff; border: 2px solid #44ccff;
    box-shadow: 0 0 15px #00aaff66;
    transform: translate(-50%, -50%); display: none;
  }
  #accuracy-result {
    font-size: 48px; font-weight: 700; margin: 20px 0;
  }
  .accuracy-excellent { color: #00ff88; }
  .accuracy-good { color: #ffaa00; }
  .accuracy-poor { color: #ff4444; }
  .val-btn {
    padding: 12px 32px; font-size: 16px; border: none; border-radius: 8px;
    cursor: pointer; font-weight: 600; margin: 8px; transition: all 0.2s;
  }
  .val-btn:hover { transform: scale(1.05); }
  #btn-recalibrate { background: #444; color: #ccc; }
  #btn-start-tracking { background: #00cc66; color: #111; }

  /* Screen 4: Tracking */
  #tracking-screen { cursor: none; }
  #gaze-dot {
    position: absolute; width: 30px; height: 30px;
    background: radial-gradient(circle, #00ff88, #00cc66);
    border-radius: 50%; transform: translate(-50%, -50%);
    box-shadow: 0 0 20px #00ff88, 0 0 40px #00ff8844;
    pointer-events: none; z-index: 100;
    transition: left 0.05s linear, top 0.05s linear;
  }
  #trail-canvas { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  #crosshair-h, #crosshair-v { position: absolute; background: #ffffff11; }
  #crosshair-h { width: 100%; height: 1px; top: 50%; }
  #crosshair-v { height: 100%; width: 1px; left: 50%; }
  #tracking-status {
    position: fixed; top: 20px; left: 20px; color: #888;
    font-family: monospace; font-size: 14px; z-index: 200; pointer-events: none;
  }
  #btn-recal-tracking {
    position: fixed; top: 20px; right: 20px; z-index: 200;
    padding: 8px 20px; font-size: 13px;
    background: #333; color: #aaa; border: 1px solid #555; border-radius: 6px;
    cursor: pointer; transition: all 0.2s;
  }
  #btn-recal-tracking:hover { background: #444; color: #fff; cursor: default; }

  /* WebGazer video overlay adjustments */
  #webgazerVideoFeed { display: none !important; }
  #webgazerFaceFeedbackBox { display: none !important; }
  #webgazerVideoCanvas { display: none !important; }
  #webgazerFaceOverlay { display: none !important; }
  .setup-active #webgazerVideoFeed {
    display: block !important; position: fixed !important;
    top: 40px !important; left: 50% !important;
    transform: translateX(-50%) !important;
    width: 320px !important; height: 240px !important;
    border-radius: 12px !important; border: 2px solid #333 !important;
    z-index: 10 !important;
  }
  .setup-active #webgazerFaceFeedbackBox {
    display: block !important; border-width: 3px !important;
    z-index: 11 !important;
  }
  .setup-active #webgazerFaceOverlay {
    display: block !important; z-index: 11 !important;
  }
  .setup-active #webgazerVideoCanvas {
    display: block !important; z-index: 11 !important;
  }
  .tracking-active #webgazerVideoFeed {
    display: block !important; position: fixed !important;
    bottom: 10px !important; right: 10px !important;
    left: auto !important; top: auto !important; transform: none !important;
    width: 160px !important; height: 120px !important;
    border-radius: 8px !important; border: 1px solid #333 !important;
    opacity: 0.6 !important; z-index: 200 !important;
  }
</style>
</head>
<body>

<!-- Screen 1: Face Setup -->
<div id="setup-screen" class="screen active">
  <div class="instructions" style="margin-top: 300px;">
    <h1>GameSense Eye Tracker</h1>
    <p>This tool tracks your eye movements using your webcam.<br>
       Allow camera access when prompted, then position your face in the frame above.</p>
    <div id="face-status" class="face-status face-bad">Initializing camera...</div>
    <button id="begin-calibration" disabled>Begin Calibration</button>
    <button id="use-previous">Use Previous Calibration</button>
  </div>
</div>

<!-- Screen 2: Calibration -->
<div id="calibration-screen" class="screen">
  <div class="cal-header">
    <h2>Calibration</h2>
    <p>Click each orange point 5 times while looking at it</p>
  </div>
  <div id="cal-points-container"></div>
  <div id="cal-progress">0 / 9 points complete</div>
</div>

<!-- Screen 3: Validation -->
<div id="validation-screen" class="screen">
  <div class="val-point" id="val-point"></div>
  <div class="val-content" id="val-content" style="display:none;">
    <h2>Calibration Quality</h2>
    <div id="accuracy-result"></div>
    <p id="accuracy-desc"></p>
    <div style="margin-top: 20px;">
      <button class="val-btn" id="btn-recalibrate">Recalibrate</button>
      <button class="val-btn" id="btn-start-tracking">Start Tracking</button>
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
  <button id="btn-recal-tracking">Recalibrate</button>
</div>

<script>
(function() {
  // ---- State ----
  let faceDetected = false;
  let currentScreen = 'setup';

  // ---- Screen management ----
  function showScreen(name) {
    currentScreen = name;
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(name + '-screen').classList.add('active');
    document.body.className = '';
    if (name === 'setup') document.body.classList.add('setup-active');
    if (name === 'tracking') document.body.classList.add('tracking-active');
  }

  // ---- WebGazer initialization ----
  async function initWebGazer() {
    webgazer.params.showVideoPreview = true;
    webgazer
      .setRegression('ridge')
      .applyKalmanFilter(true)
      .showVideoPreview(true)
      .showPredictionPoints(false)
      .showFaceOverlay(true)
      .showFaceFeedbackBox(true);

    await webgazer.begin();
    webgazer.showVideoPreview(true);
    webgazer.showFaceFeedbackBox(true);
    webgazer.showFaceOverlay(true);

    // Poll for face detection
    setInterval(checkFace, 500);

    // Check for previous calibration data
    checkPreviousCalibration();
  }

  function checkFace() {
    const prediction = webgazer.getCurrentPrediction();
    const statusEl = document.getElementById('face-status');
    const btn = document.getElementById('begin-calibration');
    if (prediction && prediction.x !== null && prediction.y !== null) {
      if (!faceDetected) {
        faceDetected = true;
        statusEl.textContent = 'Face detected - ready!';
        statusEl.className = 'face-status face-ok';
        btn.disabled = false;
      }
    } else {
      if (faceDetected || statusEl.textContent === 'Initializing camera...') {
        faceDetected = false;
        statusEl.textContent = 'No face detected - adjust position';
        statusEl.className = 'face-status face-bad';
        if (currentScreen === 'setup') btn.disabled = true;
      }
    }
  }

  function checkPreviousCalibration() {
    // WebGazer stores data in localforage/IndexedDB. If data exists, offer reuse.
    if (window.localforage) {
      localforage.getItem('webgazerGlobalData').then(data => {
        if (data) {
          document.getElementById('use-previous').style.display = 'inline-block';
        }
      }).catch(() => {});
    } else {
      // Try checking after a delay for localforage to load
      setTimeout(() => {
        if (window.localforage) {
          localforage.getItem('webgazerGlobalData').then(data => {
            if (data) {
              document.getElementById('use-previous').style.display = 'inline-block';
            }
          }).catch(() => {});
        }
      }, 2000);
    }
  }

  // ---- Calibration ----
  const CAL_POSITIONS = [
    [10, 10], [50, 10], [90, 10],
    [10, 50], [50, 50], [90, 50],
    [10, 90], [50, 90], [90, 90]
  ];
  const CLICKS_PER_POINT = 5;
  let calOrder = [];
  let calIndex = 0;
  let calClicks = {};

  function shuffleArray(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function startCalibration() {
    // Pause gaze listener during calibration so only clicks train the model
    webgazer.clearGazeListener();
    showScreen('calibration');

    calOrder = shuffleArray([0, 1, 2, 3, 4, 5, 6, 7, 8]);
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
      const idx = calOrder[calIndex];
      document.getElementById('cal-pt-' + idx).className = 'cal-point active';
    }
    updateCalProgress();
  }

  function onCalClick(pointIndex) {
    if (calOrder[calIndex] !== pointIndex) return; // only accept clicks on active point

    calClicks[pointIndex]++;
    const pt = document.getElementById('cal-pt-' + pointIndex);
    pt.textContent = calClicks[pointIndex] + '/' + CLICKS_PER_POINT;

    if (calClicks[pointIndex] >= CLICKS_PER_POINT) {
      pt.className = 'cal-point done';
      pt.textContent = '\u2713';
      calIndex++;
      if (calIndex >= 9) {
        finishCalibration();
        return;
      }
      highlightCalPoint();
    }
  }

  function updateCalProgress() {
    const done = Object.values(calClicks).filter(c => c >= CLICKS_PER_POINT).length;
    document.getElementById('cal-progress').textContent = done + ' / 9 points complete';
  }

  function finishCalibration() {
    // Brief pause to let regression stabilize
    document.getElementById('cal-progress').textContent = 'Processing...';
    setTimeout(() => {
      startValidation();
    }, 600);
  }

  // ---- Validation ----
  const VAL_POSITIONS = [
    [30, 30], [70, 30], [50, 50], [30, 70], [70, 70]
  ];
  let valIndex = 0;
  let valErrors = [];
  let valSamples = [];
  let valTimer = null;

  function startValidation() {
    showScreen('validation');
    document.getElementById('val-content').style.display = 'none';
    document.getElementById('val-status').style.display = 'block';
    valIndex = 0;
    valErrors = [];
    showValPoint();
  }

  function showValPoint() {
    if (valIndex >= VAL_POSITIONS.length) {
      finishValidation();
      return;
    }

    document.getElementById('val-progress-text').textContent =
      'Point ' + (valIndex + 1) + ' / ' + VAL_POSITIONS.length;

    const pos = VAL_POSITIONS[valIndex];
    const pt = document.getElementById('val-point');
    pt.style.left = pos[0] + 'vw';
    pt.style.top = pos[1] + 'vh';
    pt.style.display = 'block';

    const targetX = pos[0] / 100 * window.innerWidth;
    const targetY = pos[1] / 100 * window.innerHeight;

    valSamples = [];
    let sampleCount = 0;

    // Collect samples for 2 seconds
    webgazer.setGazeListener((data, timestamp) => {
      if (data && sampleCount < 50) {
        valSamples.push({ x: data.x, y: data.y });
        sampleCount++;
      }
    });

    setTimeout(() => {
      webgazer.clearGazeListener();
      pt.style.display = 'none';

      if (valSamples.length > 5) {
        const avgX = valSamples.reduce((s, p) => s + p.x, 0) / valSamples.length;
        const avgY = valSamples.reduce((s, p) => s + p.y, 0) / valSamples.length;
        const error = Math.sqrt(Math.pow(avgX - targetX, 2) + Math.pow(avgY - targetY, 2));
        valErrors.push(error);
      }

      valIndex++;
      setTimeout(showValPoint, 300);
    }, 2000);
  }

  function finishValidation() {
    document.getElementById('val-status').style.display = 'none';
    document.getElementById('val-content').style.display = 'block';

    const avgError = valErrors.length > 0
      ? valErrors.reduce((a, b) => a + b, 0) / valErrors.length
      : 999;

    // Convert pixel error to a percentage of screen diagonal
    const diag = Math.sqrt(window.innerWidth ** 2 + window.innerHeight ** 2);
    const errorPct = (avgError / diag) * 100;
    const accuracy = Math.max(0, Math.round(100 - errorPct * 3));

    const resultEl = document.getElementById('accuracy-result');
    const descEl = document.getElementById('accuracy-desc');

    resultEl.textContent = accuracy + '%';
    if (accuracy >= 80) {
      resultEl.className = 'accuracy-excellent';
      descEl.textContent = 'Excellent accuracy! Ready for tracking.';
    } else if (accuracy >= 50) {
      resultEl.className = 'accuracy-good';
      descEl.textContent = 'Good accuracy. Recalibration may improve results.';
    } else {
      resultEl.className = 'accuracy-poor';
      descEl.textContent = 'Low accuracy. Consider recalibrating with better lighting.';
    }

    // Report to server
    fetch('/calibration_status', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ accuracy: accuracy, avgErrorPx: Math.round(avgError) })
    }).catch(() => {});
  }

  // ---- Tracking ----
  let emaX = null, emaY = null;
  const EMA_ALPHA = 0.3;
  let latestGaze = null;
  let lastFaceTime = 0;
  let trailCtx = null;
  let prevTrailX = null, prevTrailY = null;

  function startTracking() {
    showScreen('tracking');

    const canvas = document.getElementById('trail-canvas');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    trailCtx = canvas.getContext('2d');
    prevTrailX = null;
    prevTrailY = null;
    emaX = null;
    emaY = null;

    window.addEventListener('resize', () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    });

    webgazer.showVideoPreview(true);
    webgazer.showFaceFeedbackBox(false);
    webgazer.showFaceOverlay(false);
    webgazer.showPredictionPoints(false);

    webgazer.setGazeListener((data, timestamp) => {
      if (data) {
        latestGaze = { x: data.x, y: data.y, ts: Date.now() };
        lastFaceTime = Date.now();
      }
    });

    // Render loop
    requestAnimationFrame(renderTracking);

    // Data send loop at 20 Hz
    setInterval(sendGazeToServer, 50);
  }

  function renderTracking() {
    if (currentScreen !== 'tracking') return;

    const dot = document.getElementById('gaze-dot');
    const status = document.getElementById('tracking-status');
    const now = Date.now();

    if (latestGaze && (now - latestGaze.ts) < 2000) {
      // EMA smoothing for display only
      if (emaX === null) {
        emaX = latestGaze.x;
        emaY = latestGaze.y;
      } else {
        emaX = EMA_ALPHA * latestGaze.x + (1 - EMA_ALPHA) * emaX;
        emaY = EMA_ALPHA * latestGaze.y + (1 - EMA_ALPHA) * emaY;
      }

      // Clamp to viewport
      const px = Math.max(0, Math.min(window.innerWidth, emaX));
      const py = Math.max(0, Math.min(window.innerHeight, emaY));

      dot.style.left = px + 'px';
      dot.style.top = py + 'px';
      dot.style.opacity = '1';

      const normX = (px / window.innerWidth).toFixed(4);
      const normY = (py / window.innerHeight).toFixed(4);
      status.textContent = 'x: ' + normX + '  y: ' + normY;
      status.style.color = '#00ff88';

      // Trail
      if (trailCtx && prevTrailX !== null) {
        trailCtx.strokeStyle = 'rgba(0,255,136,0.15)';
        trailCtx.lineWidth = 2;
        trailCtx.beginPath();
        trailCtx.moveTo(prevTrailX, prevTrailY);
        trailCtx.lineTo(px, py);
        trailCtx.stroke();
      }
      trailCtx.fillStyle = 'rgba(26,26,46,0.02)';
      trailCtx.fillRect(0, 0, trailCtx.canvas.width, trailCtx.canvas.height);
      prevTrailX = px;
      prevTrailY = py;
    } else {
      // Face lost
      if (now - lastFaceTime > 1000) {
        status.textContent = 'Face not detected';
        status.style.color = '#ff4444';
        dot.style.opacity = '0.3';
        prevTrailX = null;
        prevTrailY = null;
      }
    }

    requestAnimationFrame(renderTracking);
  }

  function sendGazeToServer() {
    if (!latestGaze || currentScreen !== 'tracking') return;
    fetch('/gaze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        x: latestGaze.x,
        y: latestGaze.y,
        screen_width: window.innerWidth,
        screen_height: window.innerHeight,
        ts: latestGaze.ts
      })
    }).catch(() => {});
  }

  function recalibrate() {
    webgazer.clearGazeListener();
    latestGaze = null;
    emaX = null;
    emaY = null;
    webgazer.clearData();
    startCalibration();
  }

  // ---- Event listeners ----
  document.getElementById('begin-calibration').addEventListener('click', () => {
    startCalibration();
  });

  document.getElementById('use-previous').addEventListener('click', () => {
    startTracking();
  });

  document.getElementById('btn-recalibrate').addEventListener('click', () => {
    recalibrate();
  });

  document.getElementById('btn-start-tracking').addEventListener('click', () => {
    startTracking();
  });

  document.getElementById('btn-recal-tracking').addEventListener('click', () => {
    recalibrate();
  });

  // ---- Init ----
  initWebGazer().catch(err => {
    console.error('WebGazer init failed:', err);
    document.getElementById('face-status').textContent = 'Camera error: ' + err.message;
  });
})();
</script>
</body>
</html>"""


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/gaze', methods=['GET'])
def get_gaze():
    with gaze_lock:
        return Response(json.dumps(latest_gaze), mimetype='application/json')


@app.route('/gaze', methods=['POST'])
def post_gaze():
    global latest_gaze
    data = request.get_json(silent=True)
    if not data:
        return Response('{"error":"no data"}', status=400, mimetype='application/json')

    entry = {
        "x": float(data.get("x", 0)),
        "y": float(data.get("y", 0)),
        "screen_width": int(data.get("screen_width", 0)),
        "screen_height": int(data.get("screen_height", 0)),
        "ts": float(data.get("ts", time.time() * 1000))
    }

    with gaze_lock:
        latest_gaze = {
            "x": entry["x"],
            "y": entry["y"],
            "tracking": True,
            "ts": entry["ts"]
        }
        gaze_buffer.append(entry)

    return Response('{"ok":true}', mimetype='application/json')


@app.route('/gaze_history')
def gaze_history():
    n = request.args.get('n', 100, type=int)
    with gaze_lock:
        items = list(gaze_buffer)[-n:]
    return Response(json.dumps(items), mimetype='application/json')


@app.route('/calibration_status', methods=['POST'])
def calibration_status():
    data = request.get_json(silent=True)
    if data:
        print(f"[CALIBRATION] Accuracy: {data.get('accuracy')}%, Avg error: {data.get('avgErrorPx')}px")
    return Response('{"ok":true}', mimetype='application/json')


if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser")
    app.run(host='0.0.0.0', port=PORT, debug=False)
