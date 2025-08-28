import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

st.set_page_config(page_title="Deforestation Monitor", page_icon="🌳", layout="centered")

# ---------- Constants ----------
STAGE1_CLASSES = ["Forest", "Deforestation"]
STAGE2_CLASSES = ["UrbanExpansion", "Infrastructure", "Agriculture", "OtherNatural"]

STAGE1_PATH = "efficientnet_b0_finalstage1.keras"
STAGE2_PATH = "efficientnet_b0_finalStage2.keras"

IMG_SIZE = (224, 224)

# ---------- Cached loaders ----------
@st.cache_resource
def load_models():
    s1 = load_model(STAGE1_PATH)
    s2 = load_model(STAGE2_PATH)
    return s1, s2

# Safe checker for model availability
def models_ready():
    try:
        _ = load_models()
        return True
    except Exception as e:
        st.error(f"Model load issue: {e}")
        return False

# ---------- Utilities ----------
def read_image(file) -> Image.Image:
    if isinstance(file, (bytes, bytearray)):
        return Image.open(io.BytesIO(file)).convert("RGB")
    return Image.open(file).convert("RGB")

def preprocess_pil(pil_img: Image.Image) -> np.ndarray:
    img = pil_img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def softmax_to_dict(scores: np.ndarray, labels: list[str]) -> dict:
    scores = scores.astype(float).tolist()
    return {labels[i]: float(scores[i]) for i in range(len(labels))}

def predict_pipeline(pil_img: Image.Image):
    (stage1, stage2) = load_models()
    x = preprocess_pil(pil_img)

    s1_scores = stage1.predict(x, verbose=0)[0]
    s1_idx = int(np.argmax(s1_scores))
    s1_label = STAGE1_CLASSES[s1_idx]
    s1_conf = float(s1_scores[s1_idx])

    if s1_label == "Forest":
        return {
            "stage1": {"label": s1_label, "confidence": s1_conf, "probs": softmax_to_dict(s1_scores, STAGE1_CLASSES)},
            "final": {"label": "Forest", "explain": "Stage 1 predicts Forest"}
        }

    # If deforestation → Stage 2
    s2_scores = stage2.predict(x, verbose=0)[0]
    s2_idx = int(np.argmax(s2_scores))
    s2_label = STAGE2_CLASSES[s2_idx]
    s2_conf = float(s2_scores[s2_idx])

    return {
        "stage1": {"label": s1_label, "confidence": s1_conf, "probs": softmax_to_dict(s1_scores, STAGE1_CLASSES)},
        "stage2": {"label": s2_label, "confidence": s2_conf, "probs": softmax_to_dict(s2_scores, STAGE2_CLASSES)},
        "final": {"label": f"Deforestation → {s2_label}", "explain": "Stage 2 refines deforestation type"}
    }

# ---------- Sidebar Navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Login", "Upload & Predict"])

# ---------- Session state for auth ----------
if "logged_in" in st.session_state:
    pass
else:
    st.session_state.logged_in = False

# ---------- Home ----------
if page == "Home":
    st.title("🌳 EuroSAT Deforestation Monitoring")
    st.write(
        """
        Two-stage pipeline with **EfficientNetB0**:
        1) Stage 1 → *Forest* vs *Deforestation*  
        2) Stage 2 → *UrbanExpansion*, *Infrastructure*, *Agriculture*, *OtherNatural*  
        """
    )
    st.info("Place your trained model files beside this script:\n"
            f"- `{STAGE1_PATH}`\n- `{STAGE2_PATH}`")

    if models_ready():
        st.success("Models are ready for inference.")

# ---------- Login ----------
elif page == "Login":
    st.title("🔐 Login")
    with st.form("login_form", clear_on_submit=False):
        user = st.text_input("Username", value="")
        pwd = st.text_input("Password", type="password", value="")
        submit = st.form_submit_button("Sign in")
    if submit:
        # Simple demo credentials — change for your deployment
        if user == "admin" and pwd == "1234":
            st.session_state.logged_in = True
            st.success("Signed in successfully.")
        else:
            st.error("Invalid credentials.")

# ---------- Upload & Predict ----------
elif page == "Upload & Predict":
    st.title("📤 Upload & Predict")

    if st.session_state.logged_in is False:
        st.warning("Please sign in on the **Login** page.")
    else:
        up = st.file_uploader("Upload a satellite image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        cam = st.camera_input("Or capture from camera (optional)")

        selected = up if up else cam
        if selected:
            pil_img = read_image(selected)
            st.image(pil_img, caption="Input", use_container_width=True)

            if models_ready():
                with st.spinner("Running inference..."):
                    result = predict_pipeline(pil_img)

                # Stage 1 block
                s1 = result["stage1"]
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Stage 1")
                    st.write(f"**Prediction:** {s1['label']}")
                    st.write(f"**Confidence:** {s1['confidence']:.3f}")
                with col2:
                    st.write("**Probabilities**")
                    st.json({k: round(v, 4) for k, v in s1["probs"].items()})

                # Stage 2 block only when deforestation
                if s1["label"] == "Deforestation" and "stage2" in result:
                    s2 = result["stage2"]
                    st.divider()
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Stage 2")
                        st.write(f"**Type:** {s2['label']}")
                        st.write(f"**Confidence:** {s2['confidence']:.3f}")
                    with col4:
                        st.write("**Probabilities**")
                        st.json({k: round(v, 4) for k, v in s2["probs"].items()})

                st.divider()
                st.success(f"**Final:** {result['final']['label']}")
                st.caption(result["final"]["explain"])

