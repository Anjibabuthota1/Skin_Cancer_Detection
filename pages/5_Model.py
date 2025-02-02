import streamlit as st
import os
from PIL import Image

# Set the folder path where the images are stored
image_folder = 'model'

# Define image filenames and corresponding titles
image_files = ['1.png', '2.png', '3.png', '4.png', '5.png']
titles = [
    "The Number Of Samples For Each Class In Train",
    "Skin Lesion Class Images",
    "Confusion Matrix",
    "Training and Validation Metrics",
    "ROC Curve"
]   

# Loop through each image and display it
for i in range(len(image_files)):
    img_path = os.path.join(image_folder, image_files[i])
    img = Image.open(img_path)
    
    # Display the title
    st.markdown(f'<p style="text-align: center; color: red; font-size: 30px;">{titles[i]}</p>', unsafe_allow_html=True)
    
    # Display the image in the center
    st.image(img, caption='', use_container_width=True, output_format='PNG')
    if i != len(image_files) - 1:
        horizontal_line_color = 'black'
        st.markdown(f'<hr style="border: 1px solid {horizontal_line_color}">', unsafe_allow_html=True)   
