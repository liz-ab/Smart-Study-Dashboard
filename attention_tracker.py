"""
attention_tracker.py

Standalone desktop version — opens a local cv2 window with your live webcam
feed and an overlay of status/score. This now shares its detection logic
with the Flask app via tracker_core.FaceTracker, instead of duplicating the
EAR/head-pose code (the original had this logic copy-pasted only here, with
the web app faking equivalent numbers with Math.random()).

Run this directly for a local debug window with no browser involved:
    python attention_tracker.py
Press ESC to quit.
"""

import cv2

from tracker_core import FaceTracker

def main():
    tracker = FaceTracker()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process(frame)
        snap = tracker.snapshot()

        cv2.putText(frame, f"Status: {result['status']}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Score: {snap['focus_pct']}%", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Blinks: {snap['blinks']}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Attention Tracker", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
