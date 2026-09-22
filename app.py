"""
app.py

StudyBerry backend. Previously this only rendered the static dashboard
template. It now also owns the webcam server-side and runs the real
MediaPipe attention tracker (tracker_core.FaceTracker), replacing the
Math.random() numbers that used to live in the frontend JS.

IMPORTANT ARCHITECTURAL NOTE:
Browsers cannot run OpenCV/MediaPipe themselves, so the server has to grab
frames from the webcam and process them. That means:
  - This only works when Flask runs LOCALLY on a machine that has a webcam
    attached (exactly what this project already assumes with debug=True).
  - It is NOT a multi-user design: there is one server-side camera shared
    by whoever hits these routes. Fine for a personal local tool; would
    need per-session camera handling (or move the CV into the browser via
    MediaPipe's JS/WASM build) for anything multi-user.
  - The dashboard's <video> element is swapped for an <img> that reads an
    MJPEG stream from /video_feed, since that's how you push a live cv2
    feed into a browser without WebRTC.

Routes:
  GET  /                    the dashboard page
  GET  /video_feed          MJPEG stream of the annotated webcam feed
  POST /api/tracker/start   reset stats, start the capture/processing thread
  POST /api/tracker/stop    stop the thread, return final session stats
  GET  /api/tracker/stats   current live stats (frontend polls this every 500ms)
"""

import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from tracker_core import FaceTracker

app = Flask(__name__)

tracker = FaceTracker()
tracker_lock = threading.Lock()

_capture_thread = None
_stop_event = threading.Event()
_latest_frame = None
_frame_lock = threading.Lock()


def _capture_loop():
    """Runs in a background thread: grabs frames, runs them through the
    tracker, and stashes the latest annotated frame for /video_feed."""
    global _latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    try:
        while not _stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            with tracker_lock:
                result = tracker.process(frame)
            _annotate(frame, result)
            with _frame_lock:
                _latest_frame = frame
            time.sleep(0.03)  # ~30fps capture; the frontend only polls stats at 2/sec
    finally:
        cap.release()


def _annotate(frame, result):
    color = (0, 200, 0) if result["focused"] else (0, 0, 220)
    cv2.putText(frame, f"Status: {result['status']}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _gen_mjpeg():
    while True:
        with _frame_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.03)


@app.route('/')
def dashboard():
    return render_template('study-dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(_gen_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/tracker/start', methods=['POST'])
def tracker_start():
    global _capture_thread
    with tracker_lock:
        tracker.reset()
    _stop_event.clear()
    if _capture_thread is None or not _capture_thread.is_alive():
        _capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        _capture_thread.start()
    return jsonify({"ok": True})


@app.route('/api/tracker/stop', methods=['POST'])
def tracker_stop():
    _stop_event.set()
    with tracker_lock:
        snap = tracker.snapshot()
    return jsonify(snap)


@app.route('/api/tracker/stats')
def tracker_stats():
    with tracker_lock:
        snap = tracker.snapshot()
    return jsonify(snap)


if __name__ == '__main__':
    # debug=True is convenient for local development but should never be
    # used in a real deployment — it turns on the Werkzeug debugger, which
    # allows arbitrary code execution if the port is ever exposed.
    app.run(debug=True, threaded=True)
