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
    "https://lottie.host/4f677c9f-5315-40c5-8414-aea82192fd59/2OWcUZOu7V.json"
)
lottie_healthy = load_lottieurl(
    "https://assets10.lottiefiles.com/packages/lf20_x1gjdldd.json"
)

st.title("Skin Cancer Detection : A Hybrid Approach Combining CNNs and Explainable AI")
st_lottie(lottie_welcome, height=500, key="welcome")
# st.header("Melanoma detection at your skin images.")
