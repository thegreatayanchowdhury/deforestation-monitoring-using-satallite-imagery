import streamlit as st
import numpy as np
from PIL import Image
import io
import os
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.efficientnet import preprocess_input
# =========================
# Environment Vars
# =========================
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

    top3_s2 = sorted(zip(STAGE2_CLASSES, s2_scores), key=lambda t: t[1], reverse=True)[:3]

    return {
        "stage1": {"label": s1_label, "confidence": s1_conf, "top3": top3_s1},
        "stage2": {"label": s2_label, "confidence": s2_conf, "top3": top3_s2},
        "final": {"label": f"Deforestation → {s2_label}", "explain": "Stage 2 refines deforestation type"}
    }

# =========================
# Streamlit App Config
# =========================
st.set_page_config(
    page_title="Deforestation Monitor",
    page_icon="🌳",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stButton>button {
        border-radius: 10px;
        background-color: #2e7d32;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .team-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
    }
    .team-card {
        width: 250px;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        background: #f9f9f9;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .team-card:hover {
        transform: scale(1.05);
        box-shadow: 3px 3px 12px rgba(0,0,0,0.2);
    }
    .team-card img {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        object-fit: cover;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Login", "Prediction", "Team"])

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.title("🌳 Deforestation Monitoring Using Satellite Imagery")
    st.write("""
        ### Overview  
        This project uses a **two-stage EfficientNetB0 pipeline**:  
        1. Stage 1 → *Forest* vs *Deforestation*  
        2. Stage 2 → If Deforestation → *Industrial, Residential, Highway, etc.*
    """)
    try:
        load_models()
        st.success("✅ Models are ready for inference.")
    except Exception as e:
        st.error(f"⚠️ Model load issue: {e}")

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
    if st.session_state.logged_in is False:
        st.warning("⚠️ Please log in first on the **Login** page.")
    else:
        up_file = st.file_uploader("Upload satellite image", type=["jpg", "jpeg", "png"])
        if up_file:
            pil_img = read_image(up_file)
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Predict"):
                try:
                    result = predict_pipeline(pil_img)
                    s1 = result["stage1"]
                    st.subheader("Stage 1: Forest vs Deforestation")
                    st.write(f"**Prediction:** {s1['label']}")
                    st.progress(s1["confidence"])
                    if s1["label"] == "Deforestation":
                        s2 = result["stage2"]
                        st.subheader("Stage 2: Deforestation Type")
                        st.write(f"**Prediction:** {s2['label']}")
                        st.progress(s2["confidence"])
                    st.success(f"🌍 Final Prediction: **{result['final']['label']}**")
                    st.caption(result["final"]["explain"])
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -------------------------
# Team Page
# -------------------------
team = [
    {"name": "AYAN CHOWDHURY", "role": "Lead Developer", "img": "images/ayan.jpg", "linkedin": "https://www.linkedin.com/in/ayan-chowdhury-4b166228b/"},
    {"name": "ASHISH KUMAR", "role": "ML Engineer", "img": "images/ashish.jpg", "linkedin": "https://www.linkedin.com/in/ashish-kumar-linkedin"},
    {"name": "SUMAN CHAKRABORTY", "role": "Research & Dataset", "img": "images/suman.jpg", "linkedin": "https://www.linkedin.com/in/suman-chakraborty-linkedin"},
    {"name": "VISHNU DEV MISHRA", "role": "Research & Dataset", "img": "images/vishnu.jpg", "linkedin": "https://www.linkedin.com/in/vishnu-dev-mishra-linkedin"}
]
elif page == "Team":
    st.title("👨‍💻 Meet Our Team")
container_html = '<div class="team-container">'
for member in team:
    card_html = f"""
    <div class="team-card">
        <a href="{member['linkedin']}" target="_blank" style="text-decoration:none;color:inherit;">
            <img src="{member['img']}" alt="{member['name']}">
            <h4>{member['name']}</h4>
            <p>{member['role']}</p>
        </a>
    </div>
    """
    container_html += card_html
container_html += '</div>'

st.markdown(container_html, unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown(
    "<small>📘 Model trained on the [EuroSat Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) "
    "EuroSat: Helber et al., IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.</small>",
    unsafe_allow_html=True
)
st.markdown("<small>© 2025 AŚVA. All rights reserved.</small>", unsafe_allow_html=True)




