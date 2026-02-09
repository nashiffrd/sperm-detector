import streamlit as st

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sperm Analysis App",
    layout="wide"
)

# ---------- SESSION STATE INIT ----------
if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "prepared_video" not in st.session_state:
    st.session_state.prepared_video = None

if "tracks_df" not in st.session_state:
    st.session_state.tracks_df = None

# ---------- SIDEBAR NAV ----------
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Halaman Awal",
        "Data Loader",
        "Data Preprocessing"
    ]

if page == "Halaman Awal":
    st.title("Aplikasi Analisis Motilitas dan Morfologi Spermatozoa")

    st.markdown("""
    Aplikasi ini digunakan untuk melakukan analisis sperma berbasis video
    menggunakan tahapan preprocessing, tracking, dan analisis lanjutan.
    """)

    st.subheader("Cara Penggunaan")
    st.markdown("""
    1. Masuk ke menu **Data Loader**
    2. Upload video sperma
    3. Jalankan preprocessing dan tracking
    4. Lanjutkan ke analisis berikutnya
    """)

    if st.button("▶ Start Analisis"):
        st.session_state.page = "Data Loader"
        st.experimental_rerun()

)

elif page == "Data Loader":
    st.header("Data Loader")

    uploaded_file = st.file_uploader(
        "Upload Video Sperma",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.session_state.video_path = temp_path
        st.success("Video berhasil diupload")

    if st.session_state.video_path:
        if st.button("➡ Lanjutkan Preprocessing"):
            st.session_state.page = "Data Preprocessing"
            st.experimental_rerun()
