import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import tensorflow.keras.backend as K

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="♋",
    layout="centered",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

@st.cache_resource
def load_model_gradcam():
    return tf.keras.models.load_model("model/model_v1.h5")

st.title("Skin Cancer Detection")

pic = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

if st.button("Predict"):
    if not pic:
        st.error("Please upload an image")
    else:
        st.header("Results")
        cols = st.columns([1, 2])
        with cols[0]:
            st.image(pic, caption=pic.name, use_container_width=True)

        with cols[1]:
            labels = [
                "actinic keratosis", "basal cell carcinoma", "dermatofibroma", "melanoma", "nevus", 
                "pigmented benign keratosis", "seborrheic keratosis", "squamous cell carcinoma", "vascular lesion"
            ]

            # Define Cancer vs Non-Cancer Classification
            cancer_labels = {
                "actinic keratosis": "Cancer",
                "basal cell carcinoma": "Cancer",
                "melanoma": "Cancer",
                "squamous cell carcinoma": "Cancer",
                "dermatofibroma": "Non-Cancer",
                "nevus": "Non-Cancer",
                "pigmented benign keratosis": "Non-Cancer",
                "seborrheic keratosis": "Non-Cancer",
                "vascular lesion": "Non-Cancer"
            }
            
            model = load_model()
            img = Image.open(pic).resize((176, 176))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img)
            predicted_index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            disease = labels[predicted_index].title()
            cancer_type = cancer_labels[labels[predicted_index]]

            st.metric("Prediction", disease)
            st.metric("Cancer Type", cancer_type)
            st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Description")
        descriptions = {
            "Actinic Keratosis": "Actinic Keratosis (AK) is a common precancerous skin condition caused by prolonged exposure to ultraviolet (UV) radiation from the sun or artificial sources like tanning beds. It appears as rough, scaly patches on sun-exposed areas of the skin, such as the face, ears, neck, hands, and forearms. The affected spots may be pink, red, or brown and can sometimes feel itchy or tender. Although AKs are generally benign, they have the potential to develop into squamous cell carcinoma (SCC), a type of skin cancer. Early detection and treatment, such as cryotherapy, topical medications, or laser therapy, are important to prevent progression to cancer. Prevention strategies include using sunscreen, wearing protective clothing, and avoiding sun exposure during peak hours.",
            "Basal Cell Carcinoma": "Basal Cell Carcinoma (BCC) is the most common type of skin cancer, typically arising in sun-exposed areas such as the face, neck, and arms. It originates from the basal cells, which are found at the bottom of the epidermis. BCC often appears as a pearly or waxy bump, a flat, scaly patch, or a sore that doesn’t heal. While it grows slowly and rarely spreads (metastasizes) to other parts of the body, it can cause significant local damage if left untreated, potentially affecting nearby tissues and bones. Risk factors include prolonged UV exposure, fair skin, and a history of sunburns. Early diagnosis and treatment through methods like surgical excision, cryotherapy, or topical medications are crucial to prevent extensive tissue damage. Using sunscreen and wearing protective clothing can help lower the risk of developing BCC.",
            "Dermatofibroma": "Dermatofibroma is a common, benign skin growth that often appears as a small, firm, raised nodule on the legs, arms, or other areas. These growths are typically reddish-brown or flesh-colored and may feel hard or tender when pressed. Dermatofibromas are believed to form as a reaction to minor skin trauma, such as insect bites or injuries, and consist of fibrous tissue. They are harmless and do not require treatment unless they cause discomfort or cosmetic concerns, in which case options like surgical removal or cryotherapy may be considered. Dermatofibromas usually remain stable in size and do not develop into cancer.",
            "Melanoma": "Melanoma is a serious and aggressive form of skin cancer that develops in melanocytes, the cells responsible for producing melanin, the pigment that gives skin its color. It often appears as a new, unusual mole or a change in an existing mole, showing irregular borders, multiple colors (brown, black, red, or blue), and asymmetry. While it can occur anywhere on the body, it is most common in sun-exposed areas. Melanoma can spread (metastasize) quickly to other organs if not detected early, making prompt diagnosis and treatment critical. Risk factors include excessive UV exposure, fair skin, a history of sunburns, and a family history of melanoma. Prevention includes regular skin checks, using sunscreen, and avoiding tanning beds. Treatment may involve surgical removal, immunotherapy, targeted therapy, or chemotherapy, depending on the stage of the cancer.",
            "Nevus": "A Nevus, commonly known as a mole, is a benign growth on the skin formed by clusters of melanocytes, the cells that produce pigment. Nevi can vary in color from flesh-toned to brown or black and may be flat or raised. Most people have several nevi, and they typically develop during childhood and adolescence. While nevi are usually harmless, changes in size, shape, color, or texture could indicate potential skin cancer, such as melanoma. Regular monitoring of moles, especially atypical ones, is important for early detection of any malignant transformation. For cosmetic or medical reasons, nevi can be removed through surgical excision or laser treatment.",
            "Pigmented Benign Keratosis": "Pigmented Benign Keratosis is a common, non-cancerous skin growth that can resemble melanoma or other malignant lesions. It is characterized by a flat, brown, black, or tan patch with a scaly or rough texture, often appearing on sun-exposed areas like the face, neck, or hands. These growths are usually harmless but can be mistaken for skin cancer due to their pigmentation and irregular borders. While pigmented benign keratoses do not require treatment, they can be removed for cosmetic reasons or to rule out malignancy. Options for removal include cryotherapy, laser therapy, or surgical excision",
            "Seborrheic Keratosis": "Seborrheic Keratosis is a common, non-cancerous skin growth that typically affects older adults. It appears as a waxy, stuck-on lesion with a rough or scaly surface, ranging in color from light tan to dark brown or black. Seborrheic keratoses can develop on any part of the body, but they are most commonly found on the chest, back, shoulders, or face. While these growths are benign and do not require treatment, they can be removed for cosmetic reasons or if they become irritated or itchy. Removal methods include cryotherapy, curettage, or laser therapy.",
            "Squamous Cell Carcinoma": "Squamous Cell Carcinoma (SCC) is a common type of skin cancer that arises from squamous cells, which are found in the outer layer of the skin (epidermis). It typically appears as a firm, red nodule, a flat, scaly patch, or a sore that doesn’t heal, often on sun-exposed areas like the face, ears, neck, and hands. SCC is primarily caused by prolonged exposure to ultraviolet (UV) radiation from the sun or tanning beds. Unlike basal cell carcinoma, SCC can be more aggressive, potentially spreading (metastasizing) to nearby tissues and other parts of the body if left untreated. Early diagnosis and treatment through surgical removal, radiation therapy, or topical medications are essential to prevent complications. Preventive measures include sun protection, regular skin checks, and avoiding excessive UV exposure.",
            "Vascular Lesion": "A vascular lesion is an abnormal growth or formation of blood vessels in the skin or underlying tissues, which can result in various types of marks or discolorations. These lesions include conditions such as hemangiomas, spider veins, and port-wine stains. Vascular lesions can appear as red, purple, or blue spots or patches, and they may be flat or raised. They often develop due to an overgrowth of blood vessels or abnormalities in the way blood vessels form. While most vascular lesions are benign and not harmful, they can sometimes be a cosmetic concern or cause discomfort if they bleed or become inflamed. Treatment options, depending on the type and severity of the lesion, may include laser therapy, sclerotherapy, or surgical removal."
        }
        st.write(descriptions.get(disease, "No description available."))

        # Grad-CAM Visualization
        model1 = load_model_gradcam()
        def make_gradcam_heatmap(img, model, last_conv_layer_name):
            heatmap_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
            with tf.GradientTape() as tape:
                inputs = tf.expand_dims(img, axis=0)
                (conv_output, predictions) = heatmap_model(inputs)
                class_id = np.argmax(predictions[0])
                class_channel = predictions[:, class_id]
            grads = tape.gradient(class_channel, conv_output)
            pooled_grads = K.mean(grads, axis=(0, 1, 2))
            conv_output = conv_output[0]
            heatmap = conv_output @ pooled_grads[..., tf.newaxis]
            heatmap = heatmap.numpy().squeeze()
            heatmap = np.maximum(heatmap, 0) / heatmap.max()
            return heatmap

        def save_and_display_gradcam(img_path, heatmap):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            jet_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 0.6, jet_heatmap, 0.4, 0)
            heatmap_path = 'heatmap/heatmap.jpg'
            cv2.imwrite(heatmap_path, superimposed_img)
            return heatmap_path

        image_path = os.path.join("uploads", pic.name)
        with open(image_path, "wb") as f:
            f.write(pic.getbuffer())

        img = Image.open(image_path).convert('RGB').resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255
        last_conv_layer_name = "block_16_depthwise"
        heatmap = make_gradcam_heatmap(img, model1, last_conv_layer_name)
        heatmap_file = save_and_display_gradcam(image_path, heatmap)

        st.divider()
        col1, col2, col3 = st.columns([1, 4, 1])
        col2.image(heatmap_file, caption="Generated Grad-CAM Heatmap", use_container_width=True)
