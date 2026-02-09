import os
import cv2
from morphology_inference.roi_extractor import extract_morphology_rois
from morphology_inference.binary_preprocess import preprocess_morphology_binary
from morphology_inference.predictor_cnn import load_model, predict_morphology


def run_morphology_inference(
    tracks_csv,
    video_dir,
    model_path,
    work_dir
):
    """
    Full morphology inference pipeline:
    tracking -> ROI -> binary -> CNN
    """

    roi_dir = os.path.join(work_dir, "roi_raw")
    binary_dir = os.path.join(work_dir, "roi_binary")

    # 1. ROI extraction
    extract_morphology_rois(
        tracks_csv=tracks_csv,
        video_dir=video_dir,
        output_dir=roi_dir
    )

    # 2. Binary preprocessing
    preprocess_morphology_binary(
        roi_dir=roi_dir,
        output_dir=binary_dir
    )

    # 3. Load CNN model
    model = load_model(model_path)

    results = []
    counts = {"normal": 0, "abnormal": 0}

    for fname in os.listdir(binary_dir):
        if not fname.endswith(".png"):
            continue

        img_path = os.path.join(binary_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        label, prob = predict_morphology(model, img)
        counts[label] += 1

        results.append({
            "filename": fname,
            "label": label,
            "prob": prob
        })

    total = counts["normal"] + counts["abnormal"]
    normal_pct = (counts["normal"] / total * 100) if total > 0 else 0
    abnormal_pct = 100 - normal_pct if total > 0 else 0

    final_label = "normal" if normal_pct > 4 else "abnormal"

    return {
        "summary": {
            "normal": counts["normal"],
            "abnormal": counts["abnormal"],
            "normal_pct": normal_pct,
            "abnormal_pct": abnormal_pct,
            "final_label": final_label
        },
        "details": results
    }
