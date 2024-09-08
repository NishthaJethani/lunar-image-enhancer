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

def shading_based_enhancement(image, filter_size=45):
    shading = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    reflectance = cv2.subtract(image, shading)
    reflectance_normalized = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX)
    final_enhanced_image = cv2.addWeighted(reflectance_normalized, 0.7, shading, 0.3, 0)
    return final_enhanced_image

def apply_homomorphic_filtering(roi, low_freq=0.3, high_freq=1.5, gamma_h=1.5, gamma_l=0.5):
    enhanced_channels = []
    for i in range(roi.shape[2]):
        img_log = np.log1p(np.array(roi[:, :, i], dtype="float"))
        dft = np.fft.fft2(img_log)
        dft_shift = np.fft.fftshift(dft)
        rows, cols = roi.shape[:2]
        mask = np.ones((rows, cols), np.float32)
        crow, ccol = rows // 2, cols // 2
        for x in range(rows):
            for y in range(cols):
                distance = np.sqrt((x - crow) ** 2 + (y - ccol) ** 2)
                mask[x, y] = gamma_l + (gamma_h - gamma_l) * (1 - np.exp(-((distance ** 2) / (2 * (low_freq ** 2)))))
        dft_shift_filtered = dft_shift * mask
        dft_ishift = np.fft.ifftshift(dft_shift_filtered)
        img_back = np.fft.ifft2(dft_ishift)
        img_back = np.real(img_back)
        homomorphic_enhanced = np.expm1(img_back)
        homomorphic_enhanced = cv2.normalize(homomorphic_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        enhanced_channels.append(homomorphic_enhanced)
    enhanced_roi = cv2.merge(enhanced_channels)
    return enhanced_roi

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
    filter_size = st.sidebar.slider("Shading-Based Enhancement Filter Size", 1, 100, 45)
    low_freq = st.sidebar.slider("Homomorphic Filtering Low Frequency", 0.1, 1.0, 0.3)
    high_freq = st.sidebar.slider("Homomorphic Filtering High Frequency", 1.0, 2.0, 1.5)
    gamma_h = st.sidebar.slider("Homomorphic Filtering Gamma High", 1.0, 3.0, 1.5)
    gamma_l = st.sidebar.slider("Homomorphic Filtering Gamma Low", 0.1, 1.0, 0.5)

    if uploaded_file is not None:
        # Convert the uploaded file to a numpy array
        image = np.array(Image.open(uploaded_file))
        original_image = image.copy()

        # Show the original image
        st.image(original_image, caption="Original Image", use_column_width=True)

        if st.button("Enhance"):
            # Apply enhancements
            gamma_corrected = adjust_gamma(image, gamma=gamma_value)
            clahe_image = apply_clahe(image, clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size))
            msr_image = multi_scale_retinex(image.astype(np.float32), scales=scales)
            msr_image = cv2.normalize(msr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            shading_enhanced = shading_based_enhancement(image, filter_size=filter_size)
            homomorphic_enhanced = apply_homomorphic_filtering(image, low_freq=low_freq, high_freq=high_freq, gamma_h=gamma_h, gamma_l=gamma_l)

            # Get image dimensions
            height, width = original_image.shape[:2]

            # Display the images side by side using st.columns
            col1, col2 = st.columns(2)
            with col1:
                st.image(resize_image(original_image, width, height), caption="Original Image", use_column_width=True)
                st.image(resize_image(gamma_corrected, width, height), caption="Gamma Correction", use_column_width=True)
                st.image(resize_image(shading_enhanced, width, height), caption="Shading-Based Enhancement", use_column_width=True)
            with col2:
                st.image(resize_image(clahe_image, width, height), caption="CLAHE", use_column_width=True)
                st.image(resize_image(msr_image, width, height), caption="Multi-Scale Retinex", use_column_width=True)
                st.image(resize_image(homomorphic_enhanced, width, height), caption="Homomorphic Filtering", use_column_width=True)

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
            
            st.sidebar.download_button(
                label="Download Shading-Based Enhanced Image",
                data=cv2.imencode('.jpg', shading_enhanced)[1].tobytes(),
                file_name='shading_enhanced.jpg',
                mime='image/jpeg'
            )
            
            st.sidebar.download_button(
                label="Download Homomorphic Enhanced Image",
                data=cv2.imencode('.jpg', homomorphic_enhanced)[1].tobytes(),
                file_name='homomorphic_enhanced.jpg',
                mime='image/jpeg'
            )

if __name__ == "__main__":
    main()
