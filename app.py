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

def resize_image(image, width, height):
    image = Image.fromarray(image)
    return image.resize((width, height))


def main():
    st.title("Image Enhancement Comparison")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Sidebar parameters
    gamma_value = st.sidebar.slider("Gamma", 0.1, 3.0, 1.0)
    clip_limit = st.sidebar.slider("CLAHE Clip Limit", 0.1, 10.0, 2.0)
    tile_grid_size = st.sidebar.slider("CLAHE Tile Grid Size", 1, 16, 8)
    scales = st.sidebar.multiselect("Multi-Scale Retinex Scales", [15, 80, 250], default=[15, 80, 250])

    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        image = np.array(Image.open(uploaded_file))
        
        original_image = image.copy()

        # Show the original image in the first column
        st.image(original_image, caption="Original Image", use_column_width=True)

        # Apply enhancements and show results side by side
        if st.button("Enhance"):
            # Enhance using all techniques
            gamma_corrected = adjust_gamma(image, gamma=gamma_value)
            clahe_image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
            msr_image = multi_scale_retinex(image.astype(np.float32), scales=scales)
            msr_image = cv2.normalize(msr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Display the images side by side using st.columns
            col1, col2 = st.columns(2)
            height, width = original_image.shape[:2]
            with col1:
                
                st.image(resize_image(original_image, width, height), caption="Original Image", use_column_width=True)
                st.image(resize_image(gamma_corrected, width, height), caption="Gamma Correction", use_column_width=True)
            with col2:
                st.image(resize_image(clahe_image, width, height), caption="CLAHE", use_column_width=True)
                st.image(resize_image(msr_image, width, height), caption="Multi-Scale Retinex", use_column_width=True)

            # Allow user to download any enhanced image
            st.sidebar.download_button(
                label="Download Gamma Corrected Image",
                data=cv2.imencode('.jpg', gamma_corrected)[1].tobytes(),
                file_name='gamma_corrected.jpg',
                mime='image/jpeg'
            )

            st.sidebar.download_button(
                label="Download CLAHE Image",
                data=cv2.imencode('.jpg', clahe_image)[1].tobytes(),
                file_name='clahe_image.jpg',
                mime='image/jpeg'
            )

            st.sidebar.download_button(
                label="Download MSR Image",
                data=cv2.imencode('.jpg', msr_image)[1].tobytes(),
                file_name='msr_image.jpg',
                mime='image/jpeg'
            )

if __name__ == "__main__":
    main()
