# 🌳 Deforestation Monitoring Using Satellite Imagery

This project implements a **two-stage machine learning pipeline** to monitor deforestation using satellite imagery.  
It is deployed as an interactive **Streamlit web application** that allows users to upload satellite images and classify them into **Forest** or different types of **Deforestation** categories.

---

## 🚀 Features

- **Two-Stage Classification Pipeline**  
  1. **Stage 1** → Classifies images as **Forest** or **Deforestation**  
  2. **Stage 2** → If Deforestation, further classifies into:
     - Industrial  
     - Residential  
     - Highway  
     - Annual Crop  
     - Permanent Crop  
     - Pasture  
     - Herbaceous Vegetation  
     - River  

- **Feature Extraction with MobileNetV2**  
  Uses pretrained MobileNetV2 as a feature extractor for satellite images.  

- **Random Forest Classifiers**  
  Stage 1 and Stage 2 models are trained separately and loaded from `.pkl` files.  

- **User Authentication**  
  Secure login system using environment variables for admin credentials.  

- **Streamlit Frontend**  
  - Upload images and view predictions  
  - Visualize confidence scores with progress bars  
  - Team page with profile cards  
  - Responsive and styled with custom CSS  

---

## 🗂️ Project Structure
.
├── app.py # Main Streamlit application
├── clf_stage1.pkl # Stage 1 RandomForest model
├── clf_stage2.pkl # Stage 2 RandomForest model
├── images/ # Team member images
│ ├── ayan.jpg
│ ├── ashish.jpg
│ ├── suman.jpg
│ └── vishnu.jpg
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/deforestation-monitor.git
   cd deforestation-monitor
   
2. **Create a virtual environment (optional but recommended)**
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. **Install dependencies**
   pip install -r requirements.txt

4. **Set environment variables**
   export ADMIN_USER=your_username
   export ADMIN_PASSWORD=your_password
   On Windows (PowerShell):
     setx ADMIN_USER "your_username"
     setx ADMIN_PASSWORD "your_password"

5. **Run the app**
   streamlit run app.py

📊 Dataset

This project is trained on the EuroSat Dataset

EuroSat: Helber et al., IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

