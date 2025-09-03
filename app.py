import streamlit as st
import numpy as np
from PIL import Image
import io
import os
import base64
import joblib

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# =========================
# Environment Vars
# =========================
admin_user = os.getenv("ADMIN_USER")
admin_password = os.getenv("ADMIN_PASSWORD")

# =========================
# Constants
# =========================
STAGE1_PATH = "clf_stage1.pkl"
STAGE2_PATH = "clf_stage2.pkl"

STAGE1_CLASSES = ["Forest", "Deforestation"]
STAGE2_CLASSES = ["Industrial", "Residential", "Highway", "AnnualCrop",
                  "PermanentCrop", "Pasture", "HerbaceousVegetation", "River"]

IMG_SIZE = (128, 128)  # must match MobileNetV2 preprocessing

# =========================
# Load Models
# =========================
@st.cache_resource
def load_models():
    stage1_model = joblib.load(STAGE1_PATH)
    stage2_model = joblib.load(STAGE2_PATH)
    return stage1_model, stage2_model

# =========================
# Feature Extractor
# =========================
@st.cache_resource
def load_feature_extractor():
    return MobileNetV2(weights="imagenet", include_top=False, pooling="avg")

base_model = load_feature_extractor()

def extract_features(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to MobileNetV2 features for RandomForest input."""
    img = pil_img.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x, verbose=0)
    return features  # shape (1, n_features)

# =========================
# Prediction Pipeline
# =========================
def predict_pipeline(pil_img: Image.Image):
    stage1, stage2 = load_models()
    x = extract_features(pil_img)

    # Stage 1: Forest vs Deforestation
    s1_scores = stage1.predict_proba(x)[0]
    s1_idx = int(np.argmax(s1_scores))
    s1_label = STAGE1_CLASSES[s1_idx]
    s1_conf = float(s1_scores[s1_idx])
    top3_s1 = sorted(zip(STAGE1_CLASSES, s1_scores), key=lambda t: t[1], reverse=True)

    if s1_label == "Forest":
        return {
            "stage1": {"label": s1_label, "confidence": s1_conf, "top3": top3_s1},
            "final": {"label": "Forest", "explain": "Stage 1 predicts Forest"}
        }

    # Stage 2: Deforestation subtypes
    s2_scores = stage2.predict_proba(x)[0]
    s2_idx = int(np.argmax(s2_scores))
    s2_label = STAGE2_CLASSES[s2_idx]
    s2_conf = float(s2_scores[s2_idx])

    # Top 3 predictions
    top3_s2 = sorted(zip(STAGE2_CLASSES, s2_scores), key=lambda t: t[1], reverse=True)[:3]

    return {
        "stage1": {"label": s1_label, "confidence": s1_conf, "top3": top3_s1},
        "stage2": {"label": s2_label, "confidence": s2_conf, "top3": top3_s2},
        "final": {"label": f"Deforestation → {s2_label}",
                  "explain": "Stage 2 refines deforestation type"}
    }

# =========================
# Custom CSS
# =========================
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
.team-card {
    border: 1px solid #ddd;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    background: #f9f9f9;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s;
    margin: 10px;
    width: 220px;
    display: inline-block;
    vertical-align: top;
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
# Streamlit App Config
# =========================
st.set_page_config(
    page_title="Deforestation Monitor",
    page_icon="🌳",
    layout="wide"
)

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
        This project uses a **two-stage ML pipeline**:  
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
            pil_img = Image.open(up_file).convert("RGB")
            st.image(pil_img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Predict"):
                try:
                    result = predict_pipeline(pil_img)

                    # Stage 1
                    s1 = result["stage1"]
                    st.subheader("Stage 1: Forest vs Deforestation")
                    for label, score in s1["top3"]:
                        st.write(f"- {label}: {score*100:.2f}%")
                        st.progress(float(score))
                    st.success(f"✅ Stage 1 Prediction: **{s1['label']}**")

                    # Stage 2
                    if s1["label"] == "Deforestation":
                        s2 = result["stage2"]
                        st.subheader("Stage 2: Deforestation Type")
                        st.write("**Top 3 Predictions:**")
                        for label, score in s2["top3"]:
                            st.write(f"- {label}: {score*100:.2f}%")
                            st.progress(float(score))
                        st.success(f"🌍 Final Prediction: **{result['final']['label']}**")
                        st.caption(result["final"]["explain"])

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# -------------------------
# Team Page
# -------------------------
elif page == "Team":
    st.title("👨‍💻 Meet Our Team")

    team = [
        {"name": "AYAN CHOWDHURY", "role": "Lead Developer", "img": "images/ayan.jpg", "linkedin": "https://www.linkedin.com/in/ayan-chowdhury-4b166228b/"},
        {"name": "ASHISH KUMAR", "role": "ML Engineer", "img": "images/ashish.jpg", "linkedin": "https://www.linkedin.com/in/ashish-kumar-08902b2a9/"},
        {"name": "SUMAN CHAKRABORTY", "role": "Research & Dataset", "img": "images/suman.jpg", "linkedin": "https://www.linkedin.com/in/suman-chakraborty-9623102a1"},
        {"name": "VISHNU DEV MISHRA", "role": "Research & Dataset", "img": "images/vishnu.jpg", "linkedin": "https://www.linkedin.com/in/vishnu-dev-mishra-05b27b28b"}
    ]

    @st.cache_data
    def img_to_base64_cached(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    cards_html = ""
    for member in team:
        if os.path.exists(member["img"]):
            img_base64 = img_to_base64_cached(member["img"])
            cards_html += f"""
            <div class="team-card">
                <a href="{member['linkedin']}" target="_blank" style="text-decoration:none;color:inherit;">
                    <img src="data:image/jpeg;base64,{img_base64}" alt="{member['name']}">
                    <h4>{member['name']}</h4>
                    <p>{member['role']}</p>
                </a>
            </div>
            """
    st.markdown(cards_html, unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<small>© 2025 AŚVA. All rights reserved.</small>", unsafe_allow_html=True)

