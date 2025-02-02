import requests
import streamlit as st
from streamlit_lottie import st_lottie


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="♋",
    layout="wide",
    initial_sidebar_state="expanded",
)

lottie_health = load_lottieurl(
    "https://lottie.host/5ffc92ca-27c7-451b-87b1-565312b8c973/5RZcSNkxlB.json"
)
lottie_welcome = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_puciaact.json"
)
lottie_healthy = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)

st.title("Skin Cancer Detection and Classification")


with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.write("##")
        st.write(
            """
            Skin cancer is a type of cancer that forms in the skin cells, 
            typically due to excessive exposure to ultraviolet (UV) radiation from the sun or tanning beds. 
            The most common types of skin cancer are basal cell carcinoma (BCC), squamous cell carcinoma (SCC), 
            and melanoma, with melanoma being the most aggressive. 
            Skin cancer can manifest as unusual growths, moles, or sores that change in size, shape, or color. 
            Early detection through regular skin checks is crucial, as skin cancer is highly treatable in its early stages. 
            Prevention includes using sunscreen, wearing protective clothing, and avoiding excessive sun exposure. 
            Treatment methods vary based on the type and stage of cancer and may involve surgical removal, radiation therapy, or immunotherapy.

            Our application detects the following diseases:
            * Actinic keratosis,
            * Basal cell carcinoma,
            * Dermatofibroma,
            * Melanoma,
            * Nevus,
            * Pigmented benign keratosis,
            * Seborrheic keratosis,
            * Squamous cell carcinoma,
            * Vascular lesion.
            """
        )
    with right_column:
        st_lottie(lottie_health, height=500, key="check")