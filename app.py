import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_image_comparison import image_comparison

# Function to apply gamma correction
def adjust_gamma(image, gamma=1.5):
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

def apply_enhancements(image, sequence, params):
    for i, technique in enumerate(sequence):
        if technique == "Gamma Correction":
            image = adjust_gamma(image, gamma=params[i]["gamma"])
        elif technique == "CLAHE":
            image = apply_clahe(image, clip_limit=params[i]["clip_limit"],
                                tile_grid_size=(params[i]["tile_grid_size"], params[i]["tile_grid_size"]))
        elif technique == "Multi-Scale Retinex":
            msr_image = multi_scale_retinex(image.astype(np.float32), scales=params[i]["scales"])
            image = cv2.normalize(msr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif technique == "Shading-Based Enhancement":
            image = shading_based_enhancement(image, filter_size=params[i]["filter_size"])
        elif technique == "Homomorphic Filtering":
            image = apply_homomorphic_filtering(image, low_freq=params[i]["low_freq"],
                                                high_freq=params[i]["high_freq"],
                                                gamma_h=params[i]["gamma_h"],
                                                gamma_l=params[i]["gamma_l"])
    return image

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
    snr = mean_signal / std_noise
    return snr

def resize_image(image, width, height):
    image = Image.fromarray(image)
    return image.resize((width, height))



def return_snr(img1):
    
    # SNR Calculation for original and enhanced images
    signal_region_img1 = get_signal_region(img1)
    noise_region_img1 = get_noise_region(img1)
    snr_img1 = calculate_snr(signal_region_img1, noise_region_img1)

    return snr_img1

# Streamlit App
def main():
    st.title("Lunar Image Enhancement")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # List to store the sequence of applied techniques
    if 'techniques' not in st.session_state:
        st.session_state.techniques = []

    # Function to add a new technique to the sequence
    def add_technique():
        st.session_state.techniques.append({
            'type': st.session_state.selected_technique,
            'params': {}
        })

    # Function to remove a technique from the sequence
    def remove_technique(index):
        del st.session_state.techniques[index]

    # Sidebar to manage techniques
    with st.sidebar:
        st.header("Add New Technique")
        st.selectbox("Select Technique", ["None", "Gamma Correction", "CLAHE", "Multi-Scale Retinex", "Shading-Based Enhancement", "Homomorphic Filtering"], key="selected_technique")
        st.button("Add Technique", on_click=add_technique)

        # Display and edit existing techniques
        if st.session_state.techniques:
            st.header("Current Techniques")
            for i, technique in enumerate(st.session_state.techniques):
                with st.expander(f"Technique {i + 1}", expanded=True):
                    st.write(f"Type: {technique['type']}")
                    if technique['type'] == "Gamma Correction":
                        technique['params']['gamma'] = st.slider(f"Gamma for Technique {i + 1}", 0.1, 3.0, 1.0)
                    elif technique['type'] == "CLAHE":
                        technique['params']['clip_limit'] = st.slider(f"CLAHE Clip Limit for Technique {i + 1}", 0.1, 10.0, 2.0)
                        technique['params']['tile_grid_size'] = st.slider(f"CLAHE Tile Grid Size for Technique {i + 1}", 1, 16, 8)
                    elif technique['type'] == "Multi-Scale Retinex":
                        technique['params']['scales'] = st.multiselect(f"MSR Scales for Technique {i + 1}", [15, 80, 250], default=[15, 80, 250])
                    elif technique['type'] == "Shading-Based Enhancement":
                        technique['params']['filter_size'] = st.slider(f"Filter Size for Technique {i + 1}", 1, 100, 45)
                    elif technique['type'] == "Homomorphic Filtering":
                        technique['params']['low_freq'] = st.slider(f"Low Frequency for Technique {i + 1}", 0.1, 1.0, 0.3)
                        technique['params']['high_freq'] = st.slider(f"High Frequency for Technique {i + 1}", 1.0, 2.0, 1.5)
                        technique['params']['gamma_h'] = st.slider(f"Gamma High for Technique {i + 1}", 1.0, 3.0, 1.5)
                        technique['params']['gamma_l'] = st.slider(f"Gamma Low for Technique {i + 1}", 0.1, 1.0, 0.5)
                    st.button(f"Remove Technique {i + 1}", key=f"remove_{i}", on_click=remove_technique, args=(i,))

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        original_image = image.copy()

        if not st.button("Apply Enhancements and Compare"):
            gamma_corrected = adjust_gamma(image)
            clahe_image = apply_clahe(image)
            msr_image = multi_scale_retinex(image.astype(np.float32))
            msr_image = cv2.normalize(msr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            shading_enhanced = shading_based_enhancement(image)
            homomorphic_enhanced = apply_homomorphic_filtering(image)

            # Get image dimensions
            height, width = original_image.shape[:2]
            # Display the images side by side using st.columns
            col1, col2 = st.columns(2)
            height, width = original_image.shape[:2]
            with col1:
                snr_gamma = return_snr(gamma_corrected)
                snr_org = return_snr(original_image)
                snr_shading = return_snr(shading_enhanced)
                st.image(resize_image(original_image, width, height), caption=f"Original Image (SNR: {snr_org:.2f})", use_column_width=True)
                st.image(resize_image(gamma_corrected, width, height), caption=f"Gamma Correction (SNR: {snr_gamma:.2f})", use_column_width=True)
                st.image(resize_image(shading_enhanced, width, height), caption=f"Shading-Based Enhancement (SNR: {snr_shading:.2f})", use_column_width=True)

            with col2:
                snr_clahe = return_snr(clahe_image)
                snr_msr = return_snr(msr_image)
                snr_homomorphic = return_snr(homomorphic_enhanced)

                st.image(resize_image(clahe_image, width, height), caption=f"CLAHE (SNR: {snr_clahe:.2f})", use_column_width=True)
                st.image(resize_image(msr_image, width, height), caption=f"Multi-Scale Retinex (SNR: {snr_msr:.2f})", use_column_width=True)
                st.image(resize_image(homomorphic_enhanced, width, height), caption=f"Homomorphic Filtering (SNR: {snr_homomorphic:.2f})", use_column_width=True)

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

        else:
            enhanced_image = apply_enhancements(image.copy(), [t['type'] for t in st.session_state.techniques], [t['params'] for t in st.session_state.techniques])
            snr1 = return_snr(original_image)
            snr2 = return_snr(enhanced_image)
            image_comparison(
                img1=original_image,
                img2=enhanced_image,
                label1=f"Original SNR {snr1:.2f}",
                label2=f"Enhanced SNR{snr2:.2f}"
            )
            # Allow downloading the enhanced image
            st.sidebar.download_button(
                label="Download Enhanced Image",
                data=cv2.imencode('.jpg', enhanced_image)[1].tobytes(),
                file_name='enhanced_image.jpg',
                mime='image/jpeg'
            )

if __name__ == "__main__":
    main()
