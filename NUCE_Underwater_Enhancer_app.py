import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import concurrent.futures
import pywt

# ==========================
# NUCE Enhancement Function
# ==========================
def nuce_enhancement(img):
    img_float = img.astype(np.float32)
    blur = cv2.GaussianBlur(img_float, (5, 5), 1.0)
    mask = cv2.subtract(img_float, blur)
    enhanced = img_float + 1.5 * np.sign(mask) * np.power(np.abs(mask), 0.8)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    return enhanced

# ==========================
# Helper: Display Histogram
# ==========================
def plot_histogram(image):
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=color)
    ax.set_title('Histogram (RGB)')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="NUCE Underwater Enhancer", layout="wide")
st.title("🌊 NUCE Underwater Image & Video Enhancer")

st.sidebar.header("User Input Parameters:")
max_images = st.sidebar.number_input("Max images to process/display", min_value=1, max_value=50, value=5)
frame_stride = st.sidebar.number_input("Video frame processing stride (process 1 frame every N)", min_value=1, max_value=10, value=2)

st.write("---")
upload_type = st.radio("Select input type:", ("Image", "Video"))

# ==========================
# IMAGE ENHANCEMENT SECTION
# ==========================
if upload_type == "Image":
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for idx, file in enumerate(uploaded_files[:max_images]):
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                enhanced_img = nuce_enhancement(img)

                st.subheader(f"Image {idx + 1}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB), caption="Enhanced Image", use_container_width=True)
                with col3:
                    plot_histogram(enhanced_img)

# ==========================
# FINAL FIXED VIDEO ENHANCEMENT SECTION
# ==========================
import concurrent.futures
import time

if upload_type == "Video":
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[-1])
        temp_input.write(uploaded_video.read())
        temp_input.flush()
        temp_input.close()
        video_path = temp_input.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Unable to open video file.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0  # Default fallback
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        if duration > 300:
            st.warning("⚠️ Video exceeds 5 minutes. Please upload a shorter video.")
            cap.release()
            os.remove(video_path)
            st.stop()

        st.info(f"🎥 Loaded video: {duration:.1f}s | {frame_width}x{frame_height} | {fps:.1f} FPS")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        total_frames = len(frames)
        if total_frames == 0:
            st.error("No frames found in the uploaded video.")
            os.remove(video_path)
            st.stop()

        estimated = (total_frames / fps) * (frame_stride / 3)
        st.write(f"⏳ Estimated enhancement time: ~{estimated:.1f} seconds")

        start_time = time.time()
        progress = st.progress(0)
        enhanced_frames = [None] * total_frames

        def process_frame(i):
            try:
                return (i, nuce_enhancement(frames[i]))
            except Exception:
                return (i, frames[i])

        st.write("🚀 Enhancing video frames...")
        indices_to_process = [i for i in range(0, total_frames, frame_stride)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for idx, result in enumerate(executor.map(process_frame, indices_to_process)):
                i, enhanced = result
                enhanced_frames[i] = enhanced
                progress.progress(int((idx / len(indices_to_process)) * 100))

        # Fill skipped frames using interpolation (to preserve timing)
        for i in range(1, total_frames):
            if enhanced_frames[i] is None:
                enhanced_frames[i] = enhanced_frames[i - 1]
        progress.progress(100)

        total_time = time.time() - start_time
        st.success(f"✅ Enhancement done in {total_time:.2f} seconds!")

        # Save enhanced video with SAME FPS as original
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = temp_output.name
        temp_output.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        for f in enhanced_frames:
            out.write(f)
        out.release()

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            st.error("❌ Failed to save output video. Try again.")
        else:
            st.success("🎬 Video enhancement complete! Now playing:")
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

            st.download_button(
                label="⬇️ Download Enhanced Video",
                data=video_bytes,
                file_name="enhanced_video.mp4",
                mime="video/mp4"
            )

        os.remove(video_path)

