from morphology_inference.roi_loader import load_roi_images
from morphology_inference.morphology_ops import preprocess_binary, extract_target_sperm
from morphology_inference.predictor import (
    load_model,
    predict_morphology_cnn
)

def run_morphology_inference(
    img_dir: str,
    model_path: str,
    threshold: float = 0.5
):
    rois = load_roi_images(img_dir)
    model = load_model(model_path)

    total = 0
    normal = 0
    abnormal = 0

    details = []

    for item in rois:
        binary = preprocess_binary(item["image"])
        target = extract_target_sperm(binary)

        if target is None:
            continue

        label, prob = predict_morphology_cnn(
            model,
            target,
            threshold=threshold
        )

        total += 1
        if label == "normal":
            normal += 1
        else:
            abnormal += 1

        details.append({
            "filename": item["filename"],
            "label": label,
            "probability": float(prob)
        })

    if total == 0:
        return None

    pct_normal = normal / total * 100
    pct_abnormal = abnormal / total * 100

    status = "NORMAL" if pct_normal > 4 else "ABNORMAL"

    return {
        "status": status,
        "total": total,
        "normal": normal,
        "abnormal": abnormal,
        "pct_normal": pct_normal,
        "pct_abnormal": pct_abnormal,
        "details": details
    }
