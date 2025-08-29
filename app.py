import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.efficientnet import preprocess_input
admin_user = os.getenv("ADMIN_USER")
admin_password = os.getenv("ADMIN_PASSWORD")
# =========================
# Constants
# =========================
STAGE1_PATH = "stage1_efficientnetb0.h5"
STAGE2_PATH = "stage2_deforestation_types_efficientnetb0.h5"

STAGE1_CLASSES = ["Forest", "Deforestation"]
STAGE2_CLASSES = ["Industrial", "Residential", "Highway", "AnnualCrop",
                  "PermanentCrop", "Pasture", "HerbaceousVegetation", "River"]

IMG_SIZE = (224, 224)  # Must match model input

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    # Stage 1
    base1 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x1 = GlobalAveragePooling2D(name="gap_stage1")(base1.output)
    out1 = Dense(len(STAGE1_CLASSES), activation="softmax", name="fc_stage1")(x1)
    stage1_model = Model(base1.input, out1)
    stage1_model.load_weights(STAGE1_PATH, by_name=True)

    # Stage 2
    base2 = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x2 = GlobalAveragePooling2D(name="gap_stage2")(base2.output)
    out2 = Dense(len(STAGE2_CLASSES), activation="softmax", name="fc_stage2")(x2)
    stage2_model = Model(base2.input, out2)
    stage2_model.load_weights(STAGE2_PATH, by_name=True)

    return stage1_model, stage2_model

# =========================
# Utilities
# =========================
def read_image(file) -> Image.Image:
    if isinstance(file, (bytes, bytearray)):
        return Image.open(io.BytesIO(file)).convert("RGB")
    return Image.open(file).convert("RGB")

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def predict_pipeline(pil_img: Image.Image):
    stage1, stage2 = load_models()
    x = preprocess_pil(pil_img)

    # Stage 1
    s1_scores = stage1.predict(x, verbose=0)[0]
    s1_idx = int(np.argmax(s1_scores))
    s1_label = STAGE1_CLASSES[s1_idx]
    s1_conf = float(s1_scores[s1_idx])

    # Debug: top 3 Stage 1 classes
    top3_s1 = sorted(zip(STAGE1_CLASSES, s1_scores), key=lambda t: t[1], reverse=True)[:3]

    if s1_label == "Forest":
        return {
            "stage1": {"label": s1_label, "confidence": s1_conf, "top3": top3_s1},
            "final": {"label": "Forest", "explain": "Stage 1 predicts Forest"}
        }

    # Stage 2
    s2_scores = stage2.predict(x, verbose=0)[0]
    s2_idx = int(np.argmax(s2_scores))
    s2_label = STAGE2_CLASSES[s2_idx]
    s2_conf = float(s2_scores[s2_idx])

    # Debug: top 3 Stage 2 classes
    top3_s2 = sorted(zip(STAGE2_CLASSES, s2_scores), key=lambda t: t[1], reverse=True)[:3]

    return {
        "stage1": {"label": s1_label, "confidence": s1_conf, "top3": top3_s1},
        "stage2": {"label": s2_label, "confidence": s2_conf, "top3": top3_s2},
        "final": {"label": f"Deforestation → {s2_label}", "explain": "Stage 2 refines deforestation type"}
    }

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Deforestation Monitor", page_icon="🌳", layout="centered")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Login", "Prediction"])

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("🌳 Deforestation Monitoring Using Satellite imagery")
    st.write("""
        Two-stage with **EfficientNetB0**
        1) Stage 1 → *Forest* vs *Deforestation*  
        2) Stage 2 → If Deforestation → *Industrial, Residential, Highway, etc.*
    """)
    # st.info(f"Models to be placed in the same folder:\n- {STAGE1_PATH}\n- {STAGE2_PATH}")
    try:
        load_models()
        st.success("Models are ready for inference.")
    except Exception as e:
        st.error(f"Model load issue: {e}")

# -------------------------
# Login Page
# -------------------------
elif page == "Login":
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Sign in")
    if submit:
        if username == admin_user and password == admin_password:
            st.session_state.logged_in = True
            st.success("Signed in successfully.")
        else:
            st.error("Invalid Username/Password")

# -------------------------
# Prediction Page
# -------------------------
elif page == "Prediction":
    st.title("📤 Upload & Predict")

    if not st.session_state.logged_in:
        st.warning("Please log in first on the **Login** page.")
    else:
        up_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
        if up_file:
            pil_img = read_image(up_file)
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                try:
                    result = predict_pipeline(pil_img)

                    # Stage 1
                    s1 = result["stage1"]
                    st.subheader("Stage 1: Forest vs Deforestation")
                    st.write(f"Prediction: **{s1['label']}**")
                    st.write(f"Confidence: {s1['confidence']:.3f}")
                    st.write("Top 3 Stage 1 probabilities:")
                    for lbl, prob in s1["top3"]:
                        st.write(f"- {lbl}: {prob:.3f}")

                    # Stage 2 only if deforestation
                    if s1["label"] == "Deforestation":
                        s2 = result["stage2"]
                        st.subheader("Stage 2: Deforestation Type")
                        st.write(f"Prediction: **{s2['label']}**")
                        st.write(f"Confidence: {s2['confidence']:.3f}")
                        st.write("Top 3 Stage 2 probabilities:")
                        for lbl, prob in s2["top3"]:
                            st.write(f"- {lbl}: {prob:.3f}")

                    # Final
                    st.success(f"Final Prediction: **{result['final']['label']}**")
                    st.caption(result["final"]["explain"])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
st.markdown("---")
st.markdown(
    "<small>📘 Model trained on the [EuroSat Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) "
    "EuroSat: A novel dataset and deep learning benchmark for land use and land cover classification, Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian, IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019, IEEE.</small>",
    unsafe_allow_html=True
)

st.markdown(
    "<small>© 2025 AŚVA. All rights reserved.</small>",
    unsafe_allow_html=True
)


