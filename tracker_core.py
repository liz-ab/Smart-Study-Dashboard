"""
tracker_core.py

Shared attention-tracking logic, used by BOTH:
  - attention_tracker.py   (standalone desktop script, cv2.imshow window)
  - app.py                 (Flask server, streams to the web dashboard)

Previously this logic only existed inside attention_tracker.py, and the web
dashboard faked its numbers with Math.random(). This module is the single
real implementation so both entry points agree on what "focused" means.
"""

import time

import numpy as np
import mediapipe as mp

# Landmark indices around the left eye (MediaPipe FaceMesh topology),
# used to compute the Eye Aspect Ratio (EAR).
LEFT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.2        # EAR below this => eyes considered closed
LONG_CLOSURE_FRAMES = 6    # consecutive closed frames => counts as a "long closure"


class FaceTracker:
    """Stateful wrapper around MediaPipe FaceMesh. Feed it frames one at a
    time via process(); read aggregate stats via snapshot()."""

    def __init__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.reset()

    def reset(self):
        self.focused_frames = 0
        self.total_frames = 0
        self.blinks = 0
        self.long_closures = 0
        self.timeline = []          # rolling list of bool ("was this frame focused")
        self._consecutive_closed = 0
        self._eyes_were_closed = False
        self.start_time = time.time()
        self.status = "No Face"

    @staticmethod
    def _ear(landmarks, w, h):
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
        v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
        h_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
        if h_dist == 0:
            return 0.3  # degenerate frame; treat as "open" rather than divide by zero
        return (v1 + v2) / (2.0 * h_dist)

    @staticmethod
    def _head_direction(landmarks):
        x = landmarks[1].x  # nose tip, normalized 0-1 across frame width
        if x < 0.4:
            return "left"
        elif x > 0.6:
            return "right"
        return "center"

    def process(self, frame_bgr):
        """Feed one BGR frame in (numpy array from cv2.VideoCapture.read()).
        Updates internal counters and returns this frame's result, mainly
        so callers can draw an overlay."""
        import cv2  # imported here so this module can be unit-tested without cv2 if ever needed

        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        eyes_open = None
        direction = None
        focused = False

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear = self._ear(landmarks, w, h)
            eyes_open = ear >= EAR_THRESHOLD
            direction = self._head_direction(landmarks)
            focused = eyes_open and direction == "center"
            self.status = "Focused" if focused else "Distracted"

            # Blink = a transition from closed -> open. A long run of closed
            # frames also gets tallied separately as a "long closure" (drowsiness signal).
            if not eyes_open:
                self._consecutive_closed += 1
                self._eyes_were_closed = True
            else:
                if self._eyes_were_closed:
                    self.blinks += 1
                if self._consecutive_closed >= LONG_CLOSURE_FRAMES:
                    self.long_closures += 1
                self._consecutive_closed = 0
                self._eyes_were_closed = False

            self.total_frames += 1
            if focused:
                self.focused_frames += 1
            self.timeline.append(focused)
            if len(self.timeline) > 120:
                self.timeline.pop(0)
        else:
            self.status = "No Face"

        return {"status": self.status, "eyes_open": eyes_open, "direction": direction, "focused": focused}

    def snapshot(self):
        """Aggregate stats for a UI to poll/display."""
        elapsed_min = max((time.time() - self.start_time) / 60.0, 1e-6)
        focus_pct = round((self.focused_frames / self.total_frames) * 100) if self.total_frames else 0
        distracted_frames = self.total_frames - self.focused_frames
        distracted_min = round(elapsed_min * (distracted_frames / self.total_frames), 1) if self.total_frames else 0
        bpm = round(self.blinks / elapsed_min, 1)
        fatigue = min(100, round(elapsed_min * 1.5 + self.blinks * 0.4))

        return {
            "status": self.status,
            "total_frames": self.total_frames,
            "focused_frames": self.focused_frames,
            "focus_pct": focus_pct,
            "distracted_min": distracted_min,
            "blinks": self.blinks,
            "blinks_per_min": bpm,
            "long_closures": self.long_closures,
            "fatigue": fatigue,
            "timeline": self.timeline[-60:],
        }
