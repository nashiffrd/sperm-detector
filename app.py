import streamlit as st
import os
import tempfile

# =============================
# PIPELINES
# =============================
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from tracking.visualization import draw_locate_frame, draw_tracks
from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference


# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Sperm Analysis Dashboard",
    layout="wide"
)

MODEL_MOTILITY_PATH = "model_motility.h5"
MODEL_MORPHOLOGY_PATH = "model_morfologi.h5"


# =============================
# SIDEBAR
# =============================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload & Tracking", "Motility Analysis", "Morphology Analysis", "Main Dashboard"]
)


# =============================
# GLOBAL SESSION STATE
# =============================
if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "tracks_csv" not in st.session_state:
    st.session_state.tracks_csv = None

if "motility_result" not in st.session_state:
    st.session_state.motility_result = None

if "morphology_result" not in st.session_state:
    st.session_state.morphology_result = None


# =========================================================
# PAGE 1 — UPLOAD & TRACKING
# =========================================================
if page == "Upload & Tracking":
    st.title("Upload Video & Tracking")

    uploaded = st.file_uploader("Upload sperm video", type=["mp4", "avi"])

    if uploaded:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_video = os.path.join(tmpdir, uploaded.name)

            with open(raw_video, "wb") as f:
                f.write(uploaded.read())

            st.info("Preprocessing video...")
            processed_video = prepare_video_pipeline(raw_video)

            st.info("Running tracking...")
            tracks_csv = tracking_pipeline(processed_video)

            st.session_state.video_path = processed_video
            st.session_state.tracks_csv = tracks_csv

            st.success("Tracking completed!")

            st.subheader("Tracking Visualization")
            frame_vis = draw_locate_frame(processed_video)
            st.image(frame_vis, caption="Detection Result")

            track_vis = draw_tracks(processed_video, tracks_csv)
            st.image(track_vis, caption="Tracking Result")


# =========================================================
# PAGE 2 — MOTILITY
# =========================================================
elif page == "Motility Analysis":
    st.title("Motility Analysis")

    if st.session_state.video_path is None:
        st.warning("Please run tracking first.")
    else:
        if st.button("Run Motility Inference"):
            st.info("Running motility model...")

            result = run_motility_inference(
                video_path=st.session_state.video_path,
                tracks_csv=st.session_state.tracks_csv,
                model_path=MODEL_MOTILITY_PATH
            )

            st.session_state.motility_result = result
            st.success("Motility inference completed!")

        if st.session_state.motility_result:
            res = st.session_state.motility_result
            st.metric("PR (%)", res["PR"])
            st.metric("NP (%)", res["NP"])
            st.metric("IM (%)", res["IM"])


# =========================================================
# PAGE 3 — MORPHOLOGY
# =========================================================
elif page == "Morphology Analysis":
    st.title("Morphology Analysis")

    roi_dir = st.text_input("ROI directory path")

    if st.button("Run Morphology Inference"):
        if not os.path.exists(roi_dir):
            st.error("ROI directory not found.")
        else:
            st.info("Running morphology model...")

            result = run_morphology_inference(
                img_dir=roi_dir,
                model_path=MODEL_MORPHOLOGY_PATH
            )

            st.session_state.morphology_result = result
            st.success("Morphology inference completed!")

    if st.session_state.morphology_result:
        labels = [r["label"] for r in st.session_state.morphology_result]
        normal_pct = labels.count("normal") / len(labels) * 100
        abnormal_pct = 100 - normal_pct

        st.metric("Normal (%)", f"{normal_pct:.2f}")
        st.metric("Abnormal (%)", f"{abnormal_pct:.2f}")


# =========================================================
# PAGE 4 — MAIN DASHBOARD
# =========================================================
elif page == "Main Dashboard":
    st.title("Main Fertility Dashboard")

    col1, col2 = st.columns(2)

    # ---------- MOTILITY ----------
    with col1:
        st.subheader("Motility Classification")

        if st.session_state.motility_result:
            m = st.session_state.motility_result
            fertile = (m["PR"] + m["NP"]) > 40

            st.metric(
                "Status",
                "FERTILE ✅" if fertile else "INFERTILE ❌"
            )

            st.write(f"PR: {m['PR']:.2f}%")
            st.write(f"NP: {m['NP']:.2f}%")
            st.write(f"IM: {m['IM']:.2f}%")
        else:
            st.info("Motility result not available.")


    # ---------- MORPHOLOGY ----------
    with col2:
        st.subheader("Morphology Classification")

        if st.session_state.morphology_result:
            labels = [r["label"] for r in st.session_state.morphology_result]
            normal_pct = labels.count("normal") / len(labels) * 100
            abnormal_pct = 100 - normal_pct

            st.metric(
                "Status",
                "NORMAL ✅" if normal_pct > 4 else "ABNORMAL ❌"
            )

            st.write(f"Normal: {normal_pct:.2f}%")
            st.write(f"Abnormal: {abnormal_pct:.2f}%")
        else:
            st.info("Morphology result not available.")
