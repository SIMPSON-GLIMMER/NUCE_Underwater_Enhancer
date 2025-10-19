import streamlit as st
import cv2
import numpy as np
import pywt
import tempfile
import requests
import os
import time
import concurrent.futures
import matplotlib.pyplot as plt

# ==========================
# NUCE Enhancement Function
# ==========================
def nuce_enhancement(img):
    """
    NUCE enhancement combining Denoising, Morphological Opening, and Wavelet Sharpening
    """
    # Convert to uint8 if needed
    img = img.astype(np.uint8)

    # Step 1: Denoising
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # Step 2: Morphological Opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

    # Step 3: Wavelet Sharpening
    coeffs = pywt.dwt2(opened, 'haar')
    cA, (cH, cV, cD) = coeffs
    cH *= 1.5
    cV *= 1.5
    cD *= 1.5
    sharpened = pywt.idwt2((cA, (cH, cV, cD)), 'haar')

    enhanced = np.clip(sharpened, 0, 255).astype(np.uint8)
    return enhanced

# ==========================
# Display Histogram
# ==========================
def plot_histogram(image):
    colors = ('b', 'g', 'r')
    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0,256])
        ax.plot(hist, color=color)
    ax.set_title('Histogram (RGB)')
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

# ==========================
# Video Processing
# ==========================
def process_video(video_path, frame_stride):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Unable to open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    if duration > 300:
        st.warning("Video >5 minutes. Please use shorter video.")
        cap.release()
        return

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        st.error("No frames in video.")
        return

    progress = st.progress(0)
    enhanced_frames = [None]*total_frames

    # Parallel enhancement
    def enhance_frame(i):
        return (i, nuce_enhancement(frames[i]))

    indices = [i for i in range(0, total_frames, frame_stride)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for idx, result in enumerate(executor.map(enhance_frame, indices)):
            i, enhanced_frames[i] = result
            progress.progress(int((idx / len(indices)) * 100))

    # Fill skipped frames
    for i in range(total_frames):
        if enhanced_frames[i] is None:
            enhanced_frames[i] = frames[i]
    progress.progress(100)

    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(temp_out.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in enhanced_frames:
        out.write(f)
    out.release()

    with open(temp_out.name, "rb") as f:
        st.video(f.read())

# ==========================
# Streamlit App
# ==========================
st.set_page_config(page_title="NUCE Underwater Enhancer", layout="wide")
st.title("🌊 NUCE Underwater Image & Video Enhancer")

st.sidebar.header("User Input Parameters:")
max_images = st.sidebar.number_input("Max images to process/display", min_value=1, max_value=50, value=5)
frame_stride = st.sidebar.number_input("Video frame processing stride (process 1 frame every N)", min_value=1, max_value=10, value=2)

st.write("---")

upload_type = st.radio(
    "Select Input Type:",
    ("Upload Image", "Upload Video", "Image URL", "Video URL")
)

# ==========================
# Image Upload
# ==========================
if upload_type == "Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
        start = time.time()
        enhanced = nuce_enhancement(img)
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="Enhanced", use_column_width=True)
        plot_histogram(enhanced)
        st.success(f"Enhanced in {time.time()-start:.2f}s")

# ==========================
# Image URL
# ==========================
elif upload_type == "Image URL":
    url = st.text_input("Enter Image URL:")
    if url:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code==200:
                img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_column_width=True)
                start = time.time()
                enhanced = nuce_enhancement(img)
                st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="Enhanced", use_column_width=True)
                plot_histogram(enhanced)
                st.success(f"Enhanced in {time.time()-start:.2f}s")
        except Exception as e:
            st.error(f"Error: {e}")

# ==========================
# Video Upload
# ==========================
elif upload_type == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4","avi","mov"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_file.read())
        temp_file.close()
        process_video(temp_file.name, frame_stride)

# ==========================
# Video URL
# ==========================
elif upload_type == "Video URL":
    url = st.text_input("Enter Video URL:")
    if url:
        try:
            resp = requests.get(url, stream=True, timeout=20)
            if resp.status_code==200:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                for chunk in resp.iter_content(1024*1024):
                    temp_file.write(chunk)
                temp_file.close()
                process_video(temp_file.name, frame_stride)
        except Exception as e:
            st.error(f"Error: {e}")



