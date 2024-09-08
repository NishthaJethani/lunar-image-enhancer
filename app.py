import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
# Function to apply gamma correction
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to apply CLAHE (Adaptive Histogram Equalization)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# Function to apply Multi-Scale Retinex (MSR)
def single_scale_retinex(img, sigma):
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log1p(img) - np.log1p(blur)

def multi_scale_retinex(img, scales=[15, 80, 250]):
    msr_result = np.zeros_like(img, dtype=np.float32)
    for sigma in scales:
        msr_result += single_scale_retinex(img, sigma)
    return msr_result / len(scales)

def main():
    
    # Streamlit App
    st.title("Chandrayaan 2 OHRC Lunar Crater Dataset Enhancer")

    # Sidebar for selecting options
    st.sidebar.title("Enhancement Options")
    enhancement_type = st.sidebar.selectbox("Choose Enhancement Type", ("Gamma Correction", "CLAHE", "Multi-Scale Retinex"))

    # Sidebar parameters for each enhancement
    if enhancement_type == "Gamma Correction":
        gamma_value = st.sidebar.slider("Gamma Value", 0.1, 3.0, 1.0)

    elif enhancement_type == "CLAHE":
        clip_limit = st.sidebar.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0)
        tile_grid_size = st.sidebar.slider("Tile Grid Size", 4, 16, 8)

    elif enhancement_type == "Multi-Scale Retinex":
        scales = st.sidebar.multiselect("Retinex Scales", [15, 80, 250], default=[15, 80, 250])

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image from the dataset", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        image = np.array(Image.open(uploaded_file))
        original_image = image.copy()

        st.image(original_image, caption="Original Image", use_column_width=True)

        if st.sidebar.button("Enhance"):
            if enhancement_type == "Gamma Correction":
                enhanced_image = adjust_gamma(image, gamma=gamma_value)
            elif enhancement_type == "CLAHE":
                enhanced_image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
            elif enhancement_type == "Multi-Scale Retinex":
                enhanced_image = multi_scale_retinex(image.astype(np.float32), scales=scales)
                enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Display the enhanced image
            st.image(enhanced_image, caption=f"Enhanced Image - {enhancement_type}", use_column_width=True)

            # Allow user to download the enhanced image
            st.sidebar.download_button(
                label="Download Enhanced Image",
                data=cv2.imencode('.jpg', enhanced_image)[1].tobytes(),
                file_name='enhanced_image.jpg',
                mime='image/jpeg'
            )

if __name__ == "__main__":
    main()