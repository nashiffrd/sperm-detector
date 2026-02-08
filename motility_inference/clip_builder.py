import cv2
import numpy as np

def extract_single_clip(cap, detections, frames=32, crop=64):
    mid = len(detections) // 2
    half = frames // 2

    selected = detections[max(0, mid-half): mid+half]
    while len(selected) < frames:
        selected.append(selected[-1])

    clip = []

    for frame_idx, x, y in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            return None

        h, w = frame.shape[:2]
        x, y = int(x), int(y)
        half_crop = crop // 2
        crop_img = frame[
            max(0, y-half_crop):min(h, y+half_crop),
            max(0, x-half_crop):min(w, x+half_crop)
        ]
        crop_img = cv2.resize(crop_img, (crop, crop))
        crop_img = crop_img.astype(np.float32) / 255.0
        clip.append(crop_img)

    return np.stack(clip)