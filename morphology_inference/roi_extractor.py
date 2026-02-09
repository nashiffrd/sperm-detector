import cv2
import os
import pandas as pd


def extract_roi_from_video(
    video_path,
    tracking_csv,
    output_dir,
    roi_size=96
):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(tracking_csv)
    cap = cv2.VideoCapture(video_path)

    saved = 0

    for _, row in df.iterrows():
        frame_idx = int(row["frame"])
        x, y = int(row["x"]), int(row["y"])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        half = roi_size // 2
        roi = gray[
            y-half:y+half,
            x-half:x+half
        ]

        if roi.shape != (roi_size, roi_size):
            continue

        fname = f"sperm_{saved:04d}.png"
        cv2.imwrite(os.path.join(output_dir, fname), roi)
        saved += 1

    cap.release()
    return saved
