from motility_inference.pipeline import run_motility_inference

VIDEO_PATH = "data/video5_prep.mp4"
TRACKS_CSV = "data/final_tracks.csv"
MODEL_PATH = "models/model_motility.h5"

result = run_motility_inference(
    video_path=VIDEO_PATH,
    tracks_csv=TRACKS_CSV,
    model_path=MODEL_PATH
)

print("\n=== MOTILITY INFERENCE RESULT ===")
print("Label utama :", result["label"])
print("Detail      :", result["detail"])