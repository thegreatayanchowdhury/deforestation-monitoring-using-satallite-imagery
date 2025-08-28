import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =========================
# Load Models
# =========================
STAGE1_PATH = "efficientnet_b0_finalstage1.h5"
STAGE2_PATH = "efficientnet_b0_finalstage2.h5"

model_stage1 = load_model(STAGE1_PATH, compile=False)
model_stage2 = load_model(STAGE2_PATH, compile=False)

# =========================
# Stage 1: Forest vs Deforestation
# =========================
STAGE1_MAP = {
    "Forest": "Forest",
    "Industrial": "Deforestation",
    "Residential": "Deforestation",
    "Highway": "Deforestation",
    "AnnualCrop": "Deforestation",
    "PermanentCrop": "Deforestation",
    "Pasture": "Deforestation",
    "HerbaceousVegetation": "Deforestation",
    "River": "Deforestation",
}

# Stage 2: Sub-categories
STAGE2_MAP = {
    0: "Industrial",
    1: "Residential",
    2: "Highway",
    3: "AnnualCrop",
    4: "PermanentCrop",
    5: "Pasture",
    6: "HerbaceousVegetation",
    7: "River",
}

# =========================
# Utility Function
# =========================
def preprocess_image(image, target_size=(128, 128)):
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image):
    img_array = preprocess_image(image)

    # Stage 1 prediction
    pred1 = model_stage1.predict(img_array)
    stage1_class = "Forest" if np.argmax(pred1) == 0 else "Deforestation"

    if stage1_class == "Forest":
        return "Forest"

    # Stage 2 prediction
    pred2 = model_stage2.predict(img_array)
    stage2_class = STAGE2_MAP[np.argmax(pred2)]
    return stage2_class

# =========================
# Streamlit App
# =========================
def main():
    st.title("🌍 Deforestation Detection System")

    menu = ["Login", "Home", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
        st.subheader("Login Section")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "12345":
                st.success("Logged in as Admin")
            else:
                st.error("Invalid Username/Password")

    elif choice == "Home":
        st.subheader("Welcome to the Deforestation Detection Project")
        st.write(
            """
            This project uses **EfficientNetB0 models (.h5 format)** to detect
            whether a given satellite image represents **Forest** or **Deforestation**.
            
            - **Stage 1**: Detect Forest vs Deforestation  
            - **Stage 2**: If Deforestation → classify into Industrial, Residential, Highway, etc.
            """
        )

    elif choice == "Prediction":
        st.subheader("Upload an Image for Prediction")

        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if image_file is not None:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)

            if st.button("Predict"):
                label = predict(image_file)
                st.success(f"Prediction: **{label}**")

if __name__ == "__main__":
    main()
