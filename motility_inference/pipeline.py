import cv2
import pandas as pd
from motility_inference.casa import compute_velocity, compute_linearity
from motility_inference.clip_builder import extract_single_clip
from motility_inference.predictor import load_model, predict_clip
from motility_inference.aggregator import aggregate_predictions

def run_motility_inference(
    video_path,
    tracks_csv,
    model_path
):
    tracks = pd.read_csv(tracks_csv)
    cap = cv2.VideoCapture(video_path)
    model = load_model(model_path)

    predictions = []

    for particle, df in tracks.groupby('particle'):
        detections = list(zip(df['frame'], df['x'], df['y']))
        clip = extract_single_clip(cap, detections)
        if clip is None:
            continue
        label, _ = predict_clip(model, clip)
        predictions.append(label)

    cap.release()
    return aggregate_predictions(predictions)