# app.py
import os
import cv2
import tempfile
import streamlit as st
import pandas as pd

# ================= MODULE IMPORT =================
from preparation.pipeline import prepare_video_pipeline
from tracking.pipeline import tracking_pipeline
from tracking.visualization import draw_locate_frame, draw_tracks

from motility_inference.pipeline import run_motility_inference
from morphology_inference.pipeline import run_morphology_inference

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Sperm Analysis App",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "Halaman Awal"

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "prepared_video" not in st.session_state:
    st.session_state.prepared_video = None

if "tracks_df" not in st.session_state:
    st.session_state.tracks_df = None

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
pages = [
    "Halaman Awal",
    "Data Loader",
    "Data Preprocessing",
    "Main Dashboard"
]

st.sidebar.title("Navigasi")
st.session_state.page = st.sidebar.radio(
    "Pilih Halaman",
    pages,
    index=pages.index(st.session_state.page)
)

# =====================================================
# HALAMAN AWAL
# =====================================================
if st.session_state.page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")

    st.markdown("""
    Aplikasi ini melakukan analisis sperma berbasis video melalui tahapan:
    **preprocessing**, **tracking**, **motility inference**, dan **morphology inference**.
    """)

    st.subheader("Alur Penggunaan")
    st.markdown("""
    1. Upload video sperma  
    2. Sistem otomatis preprocessing & tracking  
    3. Visualisasi hasil tracking  
    4. Dashboard akhir motility & morphology  
    """)

    if st.button("▶ Start Analysis"):
        st.session_state.page = "Data Loader"
        st.rerun()

# =====================================================
# DATA LOADER
# =====================================================
elif st.session_state.page == "Data Loader":
    st.header("Data Loader")

    uploaded_file = st.file_uploader(
        "Upload Video Sperma",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = video_path
        st.session_state.prepared_video = None
        st.session_state.tracks_df = None

        st.success("Video berhasil diupload")

        if st.button("➡ Lanjutkan Preprocessing"):
            st.session_state.page = "Data Preprocessing"
            st.rerun()

# =====================================================
# DATA PREPROCESSING & TRACKING
# =====================================================
elif st.session_state.page == "Data Preprocessing":
    st.header("Data Preprocessing & Tracking")

    if st.session_state.video_path is None:
        st.warning("Silakan upload video terlebih dahulu.")
        st.stop()

    # ================= PREPROCESSING =================
    if st.session_state.prepared_video is None:
        with st.spinner("Menjalankan preprocessing video..."):
            work_dir = tempfile.mkdtemp()
            st.session_state.prepared_video = prepare_video_pipeline(
                input_video_path=st.session_state.video_path,
                working_dir=work_dir
            )

    # ================= TRACKING =================
    if st.session_state.tracks_df is None:
        with st.spinner("Menjalankan tracking sperma..."):
            output_csv = os.path.join(
                os.path.dirname(st.session_state.prepared_video),
                "final_tracks.csv"
            )

            tracks = tracking_pipeline(
                prepared_video_path=st.session_state.prepared_video,
                output_csv_path=output_csv
            )

            st.session_state.tracks_df = tracks.reset_index(drop=True)

    tracks_df = st.session_state.tracks_df

    # ================= INFO =================
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Deteksi", len(tracks_df))
    with col2:
        st.metric("Total Partikel", tracks_df["particle"].nunique())

    st.divider()

    # ================= VISUALIZATION =================
    cap = cv2.VideoCapture(st.session_state.prepared_video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Gagal membaca frame hasil preprocessing.")
        st.stop()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_idx = tracks_df["frame"].min()

    locate_vis = draw_locate_frame(
        frame_gray=frame_gray,
        detections_df=tracks_df,
        frame_idx=frame_idx
    )

    track_vis = draw_tracks(
        frame_gray=frame_gray,
        tracks_df=tracks_df,
        frame_idx=frame_idx
    )

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Locate Result")
        st.image(locate_vis, channels="BGR")

    with colB:
        st.subheader("Tracking Result")
        st.image(track_vis, channels="BGR")

    st.divider()
    st.subheader("Final Tracks Data")
    st.dataframe(tracks_df, use_container_width=True)

# =====================================================
# MAIN DASHBOARD
# =====================================================
elif st.session_state.page == "Main Dashboard":
    st.header("Main Analysis Dashboard")

    if st.session_state.prepared_video is None or st.session_state.tracks_df is None:
        st.warning("Selesaikan preprocessing & tracking terlebih dahulu.")
        st.stop()

    with st.spinner("Menjalankan analisis motility & morphology..."):
        # ================= MOTILITY =================
        motility_result = run_motility_inference(
            video_path=st.session_state.prepared_video,
            tracks_csv=os.path.join(
                os.path.dirname(st.session_state.prepared_video),
                "final_tracks.csv"
            ),
            model_path="model_motility.h5"
        )

        pr = motility_result["detail"]["PR"]
        np_ = motility_result["detail"]["NP"]
        im = motility_result["detail"]["IM"]

        total = pr + np_ + im
        pct_pr = pr / total * 100 if total > 0 else 0
        pct_np = np_ / total * 100 if total > 0 else 0
        pct_im = im / total * 100 if total > 0 else 0

        motility_status = "FERTIL" if (pct_pr + pct_np) > 40 else "INFERTIL"

        # ================= MORPHOLOGY =================
        morphology_result = run_morphology_inference(
            model_path="model_morfologi.h5"
        )

        pct_normal = morphology_result["pct_normal"]
        pct_abnormal = morphology_result["pct_abnormal"]
        morphology_status = (
            "NORMAL" if pct_normal > 4 else "ABNORMAL"
        )

    # ================= DASHBOARD =================
    st.markdown("## 🧬 Ringkasan Hasil Analisis")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Motility")
            st.markdown(f"### **{motility_status}**")
            st.write(f"PR : {pct_pr:.2f}%")
            st.write(f"NP : {pct_np:.2f}%")
            st.write(f"IM : {pct_im:.2f}%")

        with col2:
            st.subheader("Morphology")
            st.markdown(f"### **{morphology_status}**")
            st.write(f"Normal   : {pct_normal:.2f}%")
            st.write(f"Abnormal : {pct_abnormal:.2f}%")
