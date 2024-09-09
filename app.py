import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from itertools import permutations
import time
import streamlit as st

# Define the enhancement functions as before

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

def save_permutations(image):
    techniques = [
        ('Gamma Correction', lambda img: adjust_gamma(img, gamma=1.0)),
        ('CLAHE', lambda img: apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8))),
        ('MSR', lambda img: multi_scale_retinex(img.astype(np.float32), scales=[15, 80, 250])),
        ('Shading', lambda img: shading_based_enhancement(img)),
        ('Homomorphic', lambda img: apply_homomorphic_filtering(img))
    ]

    # Create a directory to save images if it doesn't exist
    output_dir = os.path.abspath('enhanced_images')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get signal and noise regions of the original image
    original_signal = get_signal_region(image)
    original_noise = get_noise_region(image)

    # Calculate SNR for the original image
    original_snr = calculate_snr(original_signal, original_noise)
    print(f"SNR for Original Image: {original_snr:.2f} dB")

    # Generate permutations and calculate SNR for each enhanced image
    for r in range(1, 6):  # Change the range depending on the number of techniques
        for perm in permutations(techniques, r):
            temp_image = image.copy()
            sequence = []

            try:
                for technique_name, technique_func in perm:
                    temp_image = technique_func(temp_image)
                    sequence.append(technique_name)

                # Get signal and noise regions for the enhanced image
                enhanced_signal = get_signal_region(temp_image)
                enhanced_noise = get_noise_region(temp_image)

                # Calculate SNR for the enhanced image
                enhanced_snr = calculate_snr(enhanced_signal, enhanced_noise)
                result_key = '_'.join(sequence)

                # Normalize the image to [0, 255] if needed
                if temp_image.max() > 1.0:
                    temp_image = np.clip(temp_image, 0, 255).astype(np.uint8)

                # Save the image with SNR in the title
                filename = os.path.join(output_dir, f"enhanced_{r}techniques_{result_key}_SNR_{enhanced_snr:.2f}dB.jpg").replace('\\', '/')
                success = cv2.imwrite(filename, temp_image)
                if success:
                    print(f"Image successfully saved at {filename} with SNR: {enhanced_snr:.2f} dB")
                else:
                    print(f"Failed to save image at {filename}")

            except Exception as e:
                print(f"Error applying sequence {' -> '.join(sequence)}: {e}")
                continue

def get_signal_region(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    signal_region = image[edges > 0]
    return signal_region


def get_noise_region(image):
    if len(image.shape) == 3:  # If the image has 3 channels (RGB)
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = image  # Image is already grayscale
    
    threshold_value, noise_mask = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    noise_region = grayscale_image[noise_mask > 0]
    
    return noise_region


def calculate_snr(signal_roi, noise_roi):
    mean_signal = np.mean(signal_roi)
    std_noise = np.std(noise_roi)
    if std_noise == 0:  # Avoid division by zero
        return float('inf')
    
    snr = mean_signal / std_noise
    return snr



def main():
    st.title("Image Enhancement and SNR Calculation")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        st.image(image, caption="Original Image", use_column_width=True)

        # SNR for the original image
        original_signal = get_signal_region(image)
        original_noise = get_noise_region(image)
        original_snr = calculate_snr(original_signal, original_noise)
        st.write(f"SNR for Original Image: {original_snr:.2f} dB")

        # Button to start processing
        if st.button("Apply All Permutations"):
            with st.spinner('Processing...'):
                save_permutations(image)  # Save images and display SNR
                st.success('All permutations processed and saved!')


if __name__ == "__main__":
    main()
