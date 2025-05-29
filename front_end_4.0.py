import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
import Reconstruction
from streamlit_image_coordinates import streamlit_image_coordinates  # New import
import fSFPRNL as ff
import testmodel as tt
import mod_fista_net as mfn
st.set_page_config(layout="wide")

# Add a button on the top right corner to open a PDF
pdf_url = "https://drive.google.com/file/d/10QrQ7bAWyWh5Z1NlbIGwR_gsCbDQlskP/view?usp=drive_link"  # Replace with your actual PDF URL

# Create two columns for the header and the button
header_col, button_col = st.columns([9, 1])

with header_col:
    st.markdown("<h1 style='text-align: center;'>Digital In-Line Holographic Imaging</h1>", unsafe_allow_html=True)
model = tt.load_model(r"D:\PRoject\MTP\Final_App\Model\mod_FISTA_Class.pth", num_classes=36)
with button_col:
    st.markdown(f"""
        <a href="{pdf_url}" target="_blank">
            <button style="
                background-color:#ADD8E6;
                color:white;
                padding:10px 20px;
                border:none;
                border-radius:5px;
                cursor:pointer;
                font-size:16px;
            ">
                Help
            </button>
        </a>
        """, unsafe_allow_html=True)

# Initialize all session state variables at the start
if 'disabled' not in st.session_state:
    st.session_state.disabled = False
if 'img_reconstructed' not in st.session_state:
    st.session_state.img_reconstructed = None
if 'zo' not in st.session_state:
    st.session_state.zo = None
if 'img_reconstructed_fista' not in st.session_state:
    st.session_state.img_reconstructed_fista = None
if 'object_type' not in st.session_state:
    st.session_state.object_type = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'selected_points' not in st.session_state:
    st.session_state.selected_points = []
if 'selected_points2' not in st.session_state:
    st.session_state.selected_points2 = []
if 'species' not in st.session_state:
    st.session_state.species = None

def scale_image(input_image, factor):
    width, height = input_image.size
    new_width = int(width / factor)
    new_height = int(height / factor)
    
    scaled_image = input_image.resize((new_width, new_height))
    return scaled_image

def scale_image2(input_image, factor):
    # Convert PIL image to NumPy array for processing
    img_array = np.array(input_image)
    
    # Calculate new dimensions
    height, width = img_array.shape[:2]
    new_width = int(width / factor)
    new_height = int(height / factor)
    
    # Resize image using PIL, as it handles resizing natively better than NumPy
    scaled_image = Image.fromarray(img_array).resize((new_width, new_height))
    
    # Convert resized image back to NumPy array for further processing
    scaled_array = np.array(scaled_image)
    
    return scaled_array

# Function to convert floating-point image to 8-bit format for PNG compatibility
def convert_to_8bit(img):
    img_min, img_max = img.min(), img.max()
    if img_max != img_min:  # Avoid division by zero
        img_8bit = 255 * (img - img_min) / (img_max - img_min)
    else:
        img_8bit = np.zeros_like(img)  # Just set to zero if all values are the same
    return img_8bit.astype(np.uint8)

# Function to toggle disabled styles
def toggle_disabled_style(disabled):
    if disabled:
        st.markdown("<style>button, input, select, .stRadio, .stCheckbox {pointer-events: none; opacity: 0.5;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>button, input, select, .stRadio, .stCheckbox {pointer-events: auto; opacity: 1;}</style>", unsafe_allow_html=True)

# Call toggle function based on session state
toggle_disabled_style(st.session_state.disabled)

# Function to display images in the two windows
def display_image(window_title, img, column):
    if img is not None:
        column.image(img, caption=window_title, use_column_width=True)
    else:
        column.image(np.zeros((250, 250, 3)), caption=window_title, use_column_width=True)

def create_minimum_intensity_plot(reconstruction):
    M, N = reconstruction[:, :, 0].shape
    I_mat = np.zeros((M, N))
    for mm in range(M):
        for nn in range(N):
            val = abs(np.min(reconstruction[mm, nn, 3:]))
            I_mat[mm, nn] = val
    return I_mat

# Function to load an image (including tif files)
def load_image(image_file):
    if image_file.type == 'image/tiff':
        img = tiff.imread(image_file)
        img = Image.fromarray(img)
    else:
        img = Image.open(image_file)
    return img

# Function to reset session state variables
def reset_reconstruction():
    st.session_state.img_reconstructed = None
    st.session_state.img_reconstructed_fista = None
    st.session_state.object_type = None
    st.session_state.selected_points = []
    st.session_state.selected_points2 = []

# Function for reconstruction
def recon_image(image, wavelength, pixel_size, start_z, end_z, noi, r_type, zss, ot, reconstruction_type):
    image = image.convert('L') 
    d = np.array(image, dtype=np.uint8)
    d = d.astype(np.float64)
    d = d / np.mean(d)
    flag_obj = 0
    flag_pos = 1

    mu = 1
    t = 0.0025
    muTV = 0.1
    nIter = 15
    nFGP = 3
    eps = 0.01
    wavelength_m = wavelength * 1e-9
    pixel_size_m = pixel_size * 1e-6
    start_z_m = start_z * 1e-3
    end_z_m = end_z * 1e-3
    zss_m = zss * 1e-3

    z_step = (end_z_m - start_z_m) / noi

    reconstructionCR, localDFSA = Reconstruction.AF_loop(d, wavelength_m, pixel_size_m, start_z_m, end_z_m, z_step, r_type, noi, zss_m)

    if ot == "2D":
        max_valDFSA, max_indDFSA = np.max(localDFSA), np.argmax(localDFSA)
        reconCR = np.abs(reconstructionCR[:, :, max_indDFSA])
        print(max_indDFSA,start_z_m+(z_step*max_indDFSA))
        reconCR_Final = (reconCR - np.min(reconCR)) / (np.max(reconCR) - np.min(reconCR))
        #econCR = reconCR - np.max(reconCR)
        #reconCR = reconCR / np.min(reconCR)

        if reconstruction_type == "FISTA" or reconstruction_type == "Both":
            
            #reconfI = ff.fSFPRNL(d, pixel_size_m, start_z_m+(z_step*max_indDFSA), wavelength_m, muTV, nIter, nFGP, 0.1,0.1)
            reconfI=mfn.fista_netmodel(reconCR_Final,mfn.FISTA_net,r"D:\PRoject\MTP\Final_App\Model\Mod_FISTA_Net.pth")
            reconfI = (np.abs(reconfI) - np.min(np.abs(reconfI))) / (np.max(np.abs(reconfI)) - np.min(np.abs(reconfI)))
            #reconfI = (reconfI * 255).astype(np.uint8)  # Scale to 8-bit for proper saving
            _,species_p=tt.predict_image(reconfI, model)
            if reconstruction_type == "Both":
                return reconCR_Final, reconfI, start_z_m + max_indDFSA * z_step,species_p
            return reconfI, start_z_m + max_indDFSA * z_step,species_p
        else:  # Conventional
            return reconCR_Final, start_z_m + max_indDFSA * z_step
    else:
        return reconstructionCR

def are_parameters_set():
    if not (wavelength > 0 and pixel_size > 0 and noi > 0):
        return False

    if illumination_type == "Spherical Wave":
        return (min_z < max_z) and (min_z < zss) and (max_z < zss)
    else:
        return min_z < max_z

# Image upload before parameter selection
st.markdown("<h3 style='font-weight: bold;'>Upload Hologram</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(" ", type=['png', 'jpg', 'jpeg', 'tif'], disabled=st.session_state.disabled, on_change=reset_reconstruction)

# Horizontal alignment of Illumination, Object, and Setup Parameters sections
col1_param, col2_param, col3_param = st.columns([1.2, 1, 2.2])

with col1_param:
    st.subheader("Illumination")
    illumination_type = st.radio("Select Illumination Type", ["Plane Wave", "Spherical Wave"], disabled=st.session_state.disabled, on_change=reset_reconstruction)

with col2_param:
    st.subheader("Object")
    object_type = st.radio("Select Object Type", ["2D", "3D"], disabled=st.session_state.disabled, on_change=reset_reconstruction)
    st.session_state.object_type = object_type

# Enable reconstruction type selection only if both conditions (2D and Plane Wave) are met
reconstruction_type_enabled = (object_type == "2D")

# Conditional reconstruction type selection using checkboxes in a horizontal layout
st.subheader("Reconstruction Type")
col_conventional, col_fista = st.columns(2)
with col_conventional:
    conventional_selected = st.checkbox("Conventional", value=True, disabled=not reconstruction_type_enabled, on_change=reset_reconstruction)
with col_fista:
    fista_selected = st.checkbox("Denoised Reconstruction", disabled=not reconstruction_type_enabled, on_change=reset_reconstruction)

with col3_param:
    st.subheader("Setup Parameters")
    setup_col1, setup_col2, setup_col3 = st.columns(3)

    with setup_col1:
        wavelength = st.number_input("Wavelength (nm)", min_value=0.0, step=1.0, disabled=st.session_state.disabled, on_change=reset_reconstruction)
        pixel_size = st.number_input("Pixel Size (μm)", min_value=0.0, step=0.1, disabled=st.session_state.disabled, on_change=reset_reconstruction)

    with setup_col2:
        min_z = st.number_input("Minimum z (mm)", min_value=0.0, step=0.1, disabled=st.session_state.disabled, on_change=reset_reconstruction)
        max_z = st.number_input("Maximum z (mm)", min_value=0.0, step=0.1, disabled=st.session_state.disabled, on_change=reset_reconstruction)

    with setup_col3:
        noi = st.number_input("No. of Samples in Z range", min_value=1, step=1, disabled=st.session_state.disabled, on_change=reset_reconstruction)

        if illumination_type == "Spherical Wave":
            zss = st.number_input("zss (mm)", min_value=0.0, step=0.1, disabled=st.session_state.disabled, on_change=reset_reconstruction)
        else:
            zss = 1.0

# Reconstruct button
reconstruct_button = st.button("Reconstruct")

# Image display section - moved below reconstruct button
if conventional_selected and fista_selected:
    display_cols = st.columns([5,5,5])
else:
    display_cols = st.columns([1,1])

# Load and display original image
img_original = None
if uploaded_file is not None:
    with display_cols[0]:
        img_original = load_image(uploaded_file)
        if fista_selected and conventional_selected:
            if img_original.size[0] > 450:
                factor = img_original.size[0] / 450
            else:
                factor = 1
            scaled_image = scale_image(img_original, factor)
        else:
            if img_original.size[0] > 700:
                factor = img_original.size[0] / 700
            else:
                factor = 1
            scaled_image = scale_image(img_original, factor)
        ma = streamlit_image_coordinates(scaled_image, key="TP")
        st.write("Original Hologram")

# Handle reconstruction and display
if reconstruct_button:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    elif not are_parameters_set():
        st.error("Please ensure:\n- Minimum z < Maximum z\n- For Spherical Wave: Minimum z < zss and Maximum z < zss\n- All setup parameters are set correctly.")
    else:
        st.session_state.disabled = True  # Disable inputs
        toggle_disabled_style(True)  # Apply disabled style

        # Reconstruction based on selected options
        if object_type == "2D":
            if conventional_selected and fista_selected:
                img_conventional, img_fista, zo,st.session_state.species = recon_image(
                    img_original, wavelength, pixel_size, min_z, max_z, 
                    noi, illumination_type, zss, object_type, "Both"
                )
                # display_image("Conventional Reconstruction", img_conventional, display_cols[1])
                # display_image("FISTA Reconstruction", img_fista, display_cols[1])
                st.session_state.img_reconstructed = img_conventional
                st.session_state.img_reconstructed_fista = img_fista
                st.session_state.zo = zo
            elif conventional_selected:
                img_reconstructed, zo = recon_image(
                    img_original, wavelength, pixel_size, min_z, max_z, 
                    noi, illumination_type, zss, object_type, "Conventional"
                )
                # display_image("Reconstructed Image - Conventional", img_reconstructed, display_cols[1])
                st.session_state.img_reconstructed = img_reconstructed
                st.session_state.zo = zo
            elif fista_selected:
                img_reconstructed, zo,st.session_state.species = recon_image(
                    img_original, wavelength, pixel_size, min_z, max_z, 
                    noi, illumination_type, zss, object_type, "FISTA"
                )
                # display_image("Reconstructed Image - FISTA", img_reconstructed, display_cols[1])
                st.session_state.img_reconstructed = img_reconstructed
                st.session_state.zo = zo
        else:
            img_reconstructed = recon_image(
                img_original, wavelength, pixel_size, min_z, max_z, 
                noi, illumination_type, zss, object_type, "Both"
            )
            st.session_state.img_reconstructed = img_reconstructed
        st.session_state.disabled = False  # Re-enable inputs
        toggle_disabled_style(False)  # Re-enable style

if object_type == "2D" and st.session_state.img_reconstructed is not None:
    # Convert reconstructed image to 8-bit for compatibility with PNG
    img_for_display = convert_to_8bit(st.session_state.img_reconstructed)
    
    if fista_selected and conventional_selected:
        if img_for_display.shape[1] > 450:
            factor = img_for_display.shape[1] / 450
        else:
            factor = 1
        scaled_image = scale_image2(img_for_display, factor)
    else:
        if img_for_display.shape[1] > 700:
            factor = img_for_display.shape[1] / 700
        else:
            factor = 1
        scaled_image = scale_image2(img_for_display, factor)
    # Display and capture coordinates for distance measurement
    with display_cols[1]:
        coords = streamlit_image_coordinates(scaled_image, key="distance_measure")
        if conventional_selected:
            st.write("Conventional Reconstruction")
        else:
            st.write("Denoised Reconstruction")
            if illumination_type== "Plane Wave":
                st.write(st.session_state.species)
        if coords:
            # Only update conventional points, don't clear FISTA points
            if len(st.session_state.selected_points) == 2:
                st.session_state.selected_points = []
            st.session_state.selected_points.append((coords['x'], coords['y']))

    # Calculate and display distance if two points are selected
    if len(st.session_state.selected_points) == 2:
        point1 = st.session_state.selected_points[0]
        point2 = st.session_state.selected_points[1]
        if illumination_type == "Spherical Wave":
            m = (zss * 1e-3) / st.session_state.zo
            print(m, zss, st.session_state.zo)
        else:
            m = 1
        # Calculate the Euclidean distance
        x0, y0 = point1[0] / factor, point1[1] / factor
        x1, y1 = point2[0] / factor, point2[1] / factor
        distance_px = np.sqrt((x1 - x0)**2 + (y1 - y0)**2) / m
        distance_um = distance_px * pixel_size  # Convert from pixels to micrometers
        # Display distance below the image
        if distance_um > 0:
            display_cols[1].write(f"Object size is {distance_um:.2f} μm")
        
    # Handle FISTA reconstruction display and measurements
    if fista_selected and conventional_selected:
        img_for_display2 = convert_to_8bit(st.session_state.img_reconstructed_fista)
        if img_for_display2.shape[1] > 450:
            factor = img_for_display2.shape[1] / 450
        else:
            factor = 1
        print(factor)
        scaled_image = scale_image2(img_for_display2, factor)
        # Display and capture coordinates for distance measurement
        with display_cols[2]:
            coords2 = streamlit_image_coordinates(scaled_image, key="distance_measure_fista")
            st.write("Denoised Reconstruction")
            if illumination_type== "Plane Wave":
                st.write(st.session_state.species)
            if coords2:
                # Only update FISTA points, don't clear conventional points
                if len(st.session_state.selected_points2) == 2:
                    st.session_state.selected_points2 = []
                st.session_state.selected_points2.append((coords2['x'], coords2['y']))

        # Calculate and display distance if two points are selected for FISTA
        if len(st.session_state.selected_points2) == 2:
            point1 = st.session_state.selected_points2[0]
            point2 = st.session_state.selected_points2[1]
            
            # Calculate the Euclidean distance
            x0, y0 = point1[0] / factor, point1[1] / factor
            x1, y1 = point2[0] / factor, point2[1] / factor
            distance_px = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            distance_um = distance_px * pixel_size  # Convert from pixels to micrometers
            # Display distance below the image
            if distance_um > 0:
                display_cols[2].write(f"Object size is {distance_um:.2f} μm")
                
# Display slider and minimum intensity plot button for 3D object type
if st.session_state.object_type == "3D" and st.session_state.img_reconstructed is not None:
    z_step = (max_z - min_z) / noi
    _, col_slider, _ = st.columns([1, 0.5, 0.5])
    with col_slider:
        z_slider = st.slider("Select z-slice", min_value=min_z + z_step, max_value=max_z - z_step, value=min_z + z_step, step=z_step)
    z_index = int((z_slider - min_z) / z_step)
    slice_image = np.abs(st.session_state.img_reconstructed[:, :, z_index])

    min_val, max_val = np.min(slice_image), np.max(slice_image)
    if max_val != min_val:
        slice_image = (slice_image - min_val) / (max_val - min_val)
        slice_image = (slice_image * 255).astype(np.uint8)
    else:
        slice_image = np.zeros_like(slice_image, dtype=np.uint8)
    with display_cols[1]:
        # slice_image = load_image(slice_image)
        if slice_image.shape[1] > 700:
            factor = slice_image.shape[1] / 700
        else:
            factor = 1
        scaled_image3 = scale_image2(slice_image, factor)
        # img_disp = img_original.resize((350, 350))
        mag = streamlit_image_coordinates(scaled_image3, key="TTP")
        
        st.write(f"Reconstructed Image at z-slice {z_slider}")
    
    if st.button("Minimum Intensity Plot"):
        st.session_state.disabled = True  # Disable inputs
        toggle_disabled_style(st.session_state.disabled)
        I_mat = create_minimum_intensity_plot(st.session_state.img_reconstructed)
        mipcol, _ = st.columns([1, 1])
        with mipcol:
            st.image(I_mat, caption="Minimum Intensity Plot", use_column_width=True, clamp=True, channels="GRAY")
        st.session_state.disabled = False  # Re-enable inputs
        toggle_disabled_style(st.session_state.disabled)
