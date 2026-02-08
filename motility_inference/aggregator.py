from collections import Counter

def aggregate_predictions(predictions):
    counts = Counter(predictions)
    main_label = counts.most_common(1)[0][0]

    return {
        "label": main_label,
        "detail": {
            "PR": counts.get("PR", 0),
            "NP": counts.get("NP", 0),
            "IM": counts.get("IM", 0),
        }
    }
