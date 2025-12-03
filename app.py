import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Image Processor Pro", layout="wide")

st.title("🎨 Interactive Image Processing App")
st.markdown("Upload an image, then choose an operation from the sidebar!")

# 2. Sidebar: Upload Image
st.sidebar.header("1. Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 3. Decode the image once
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1) # BGR format

    # 4. Sidebar: Main Menu
    st.sidebar.header("2. Choose Operation")
    options = [
        "Preview Original",
        "Image Properties ",
        "Grayscale & B/W ",
        "Rotate & Flip ",
        "Object & Edge Detection ",
        "Splitting & Grid "
    ]
    choice = st.sidebar.selectbox("Select a task:", options)

    # --- LOGIC BASED ON SELECTION ---

    # OPTION: Preview
    if choice == "Preview Original":
        st.subheader("Original Image Preview")
        st.image(original_image, channels="BGR", use_container_width=True)

    # OPTION: Properties
    elif choice == "Image Properties ":
        st.subheader("📊 Image Properties")
        h, w, c = original_image.shape
        size_mb = uploaded_file.size / (1024 * 1024)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Height", f"{h} px")
        c2.metric("Width", f"{w} px")
        c3.metric("Channels", c)
        
        st.info(f"File Size: {size_mb:.2f} MB")
        st.info(f"Total Pixels: {h * w:,}")
        
        st.image(original_image, channels="BGR", caption="Analyzed Image", width=400)

    # OPTION: Grayscale
    elif choice == "Grayscale & B/W ":
        st.subheader("⚫⚪ Grayscale Conversion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, channels="BGR", caption="Original", use_container_width=True)
        
        with col2:
            gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            st.image(gray_img, caption="Grayscale", use_container_width=True)

    # OPTION: Rotate & Flip
    elif choice == "Rotate & Flip ":
        st.subheader("🔄 Geometric Transformations")
        
        transform_type = st.radio("Select Transformation:", 
                                  ["Rotate 90° Clockwise", "Rotate 180°", "Rotate 90° CCW", "Mirror (Flip)"])
        
        if transform_type == "Rotate 90° Clockwise":
            processed = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
        elif transform_type == "Rotate 180°":
            processed = cv2.rotate(original_image, cv2.ROTATE_180)
        elif transform_type == "Rotate 90° CCW":
            processed = cv2.rotate(original_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif transform_type == "Mirror (Flip)":
            processed = cv2.flip(original_image, 1)

        col1, col2 = st.columns(2)
        col1.image(original_image, channels="BGR", caption="Original", use_container_width=True)
        col2.image(processed, channels="BGR", caption="Transformed", use_container_width=True)

    # OPTION: Detection
    elif choice == "Object & Edge Detection ":
        st.subheader("🕵️ Object & Edge Detection")
        
        # Add a slider to let user adjust sensitivity!
        st.write("Adjust Edge Detection Thresholds:")
        t_lower = st.slider("Lower Threshold", 0, 255, 100)
        t_upper = st.slider("Upper Threshold", 0, 255, 200)

        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, t_lower, t_upper)
        
        # Contours
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_img = original_image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        c1, c2 = st.columns(2)
        c1.image(edges, caption="Canny Edges (Adjust with slider)", use_container_width=True)
        c2.image(contour_img, channels="BGR", caption=f"Contours Found: {len(contours)}", use_container_width=True)

    # OPTION: Splitting
    elif choice == "Splitting & Grid ":
        st.subheader("✂️ Image Slicing")
        slice_type = st.selectbox("Choose slice type:", ["Vertical Split (80/20)", "Horizontal Split (70/30)", "5-Row Grid"])
        
        h, w = original_image.shape[:2]

        if slice_type == "Vertical Split (80/20)":
            split = int(0.8 * w)
            left = original_image[:, :split]
            right = original_image[:, split:]
            c1, c2 = st.columns([4, 1])
            c1.image(left, channels="BGR", caption="80%", use_container_width=True)
            c2.image(right, channels="BGR", caption="20%", use_container_width=True)
            
        elif slice_type == "Horizontal Split (70/30)":
            split = int(0.7 * h)
            top = original_image[:split, :]
            bottom = original_image[split:, :]
            st.image(top, channels="BGR", caption="Top 70%", use_container_width=True)
            st.image(bottom, channels="BGR", caption="Bottom 30%", use_container_width=True)
            
        elif slice_type == "5-Row Grid":
            rows = 5
            step = h // rows
            for i in range(rows):
                start = i * step
                end = h if i == rows - 1 else (i + 1) * step
                sub_img = original_image[start:end, :]
                st.image(sub_img, channels="BGR", caption=f"Row {i+1}")

else:
    # Initial state when no file is uploaded
    st.info("Waiting for image upload...")
