import os
import cv2
import json
import argparse
import numpy as np
import mediapipe as mp

HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay hand landmarks from JSON and draw skeleton on black background"
    )
    p.add_argument(
        "--session", "-s",
        required=True,
        help="Path to session folder containing 'coords/'"
    )
    return p.parse_args()

def load_frame_metadata(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["hands"]

def draw_hand(frame, hand_data, color):
    h, w = frame.shape[:2]
    # map id -> (x_norm, y_norm)
    pts = {lm["id"]:(lm["x"], lm["y"]) for lm in hand_data["landmarks"]}

    # draw landmarks
    for idx, (xn, yn) in pts.items():
        px, py = int(xn * w), int(yn * h)
        cv2.circle(frame, (px, py), 5, color, -1)

    # draw connections
    for start, end in HAND_CONNECTIONS:
        if start in pts and end in pts:
            x1, y1 = int(pts[start][0] * w), int(pts[start][1] * h)
            x2, y2 = int(pts[end][0]   * w), int(pts[end][1]   * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

def main():
    args = parse_args()
    coords_dir = os.path.join(args.session, "coords")

    # sorted list of JSON files by frame index
    json_files = sorted(
        os.listdir(coords_dir),
        key=lambda fn: int(fn.split("_")[0])
    )

    for jf in json_files:
        json_path = os.path.join(coords_dir, jf)

        # create a black background canvas (480×640)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # load metadata and draw hands
        hands = load_frame_metadata(json_path)
        for hand in hands:
            label = hand["handedness"]["label"]
            color = (0, 255,   0) if label == "Left" else (0, 0, 255)
            draw_hand(frame, hand, color)

        cv2.imshow("Replay Hands", frame)
        if cv2.waitKey(30) & 0xFF in (27, ord('q')):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
