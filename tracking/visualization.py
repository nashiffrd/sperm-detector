import cv2
import numpy as np
import pandas as pd

def draw_locate_frame(frame_gray, detections_df, frame_idx):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    df = detections_df[detections_df["frame"] == frame_idx]

    for _, r in df.iterrows():
        cv2.circle(
            vis,
            (int(r["x"]), int(r["y"])),
            8,
            (0, 255, 0),
            2
        )
    return vis


def draw_tracks(frame_gray, tracks_df, frame_idx):
    vis = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # warna konsisten per particle
    rng = np.random.default_rng(42)
    particle_ids = tracks_df["particle"].unique()
    colors = {
        pid: tuple(int(c) for c in rng.integers(50, 255, size=3))
        for pid in particle_ids
    }

    for pid in particle_ids:
        grp = tracks_df[tracks_df["particle"] == pid]
        grp = grp.sort_values("frame")

        pts = grp[grp["frame"] <= frame_idx][["x", "y"]].values.astype(int)

        # draw trajectory
        for i in range(1, len(pts)):
            cv2.line(
                vis,
                tuple(pts[i - 1]),
                tuple(pts[i]),
                colors[pid],
                2           # 🔥 lebih tebal seperti tp.plot_traj
            )

        # draw current position (kepala lintasan)
        if len(pts) > 0:
            cv2.circle(
                vis,
                tuple(pts[-1]),
                3,
                colors[pid],
                -1
            )

    return vis
