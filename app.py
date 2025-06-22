import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
from recommendation import cnv, dme, drusen, normal  # Make sure these variables are defined
import base64
import os

# ---------------------- Title & Logo ----------------------
import streamlit as st

# Inject custom CSS
st.markdown("""
    <style>
    /* Main app background with gradient */
            div[data-testid="stStatusWidget"] {
    background-color: #1e1e1e !important;  /* Dark gray example */
    color: white !important;               /* Text color for contrast */
    border-radius: 8px;
    padding: 6px;
}
    .stApp {
        background: linear-gradient(to right, white, #306998);
        background-attachment: fixed;
        color: black;
    }

    /* Header style */
    h1 {
        font-family: 'Segoe UI', sans-serif;
        color: black;
    }

    /* Sidebar background and text */
    section[data-testid="stSidebar"] {
        background: linear-gradient(to left, white, #306998);
        color: black;
    }

    /* Force all sidebar elements to use black font */
    section[data-testid="stSidebar"] * {
        color: black !important;
    }

    /* Disease Identification page button styling */
    .disease-button {
        background-color: #ff6347; /* Tomato red */
        color: white; /* Button text color */
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
    }
    .disease-button:hover {
        background-color: #ff4500; /* Darker red on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Logo and title aligned
col1, col2 = st.columns([1, 4])

with col1:
    st.image("assets/image.png", width=80)  # Adjust path as needed

with col2:
    st.markdown("<h1 style='margin-top: 10px;'>OCT Retinal Analysis Platform</h1>", unsafe_allow_html=True)

# ---------------------- Prediction Function ----------------------
def model_prediction(test_image_path):
    model = tf.keras.models.load_model("Trained_Eye_disease_model.keras")
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)

# ---------------------- Manage Navigation ----------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Home"

# Sidebar Navigation
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Identification"],
                            index=["Home", "About", "Disease Identification"].index(st.session_state.app_mode))

# ---------------------- Home Page ----------------------
if app_mode == "Home":
    st.title(" Welcome to the OCT Retinal Analysis Platform")
    st.markdown("""
<h3 style='color: black; text-align: center;'>
    Empowering Early Detection with AI-Powered Retinal Diagnosis
</h3>
""", unsafe_allow_html=True)


    st.image("assets/retina_banner.png", use_column_width=True, caption="AI-driven OCT Imaging for Disease Detection")
    st.markdown("""
<h2 style='color: black;'> AI-driven OCT Imaging for Disease Detection</h2>

<p style='color: black; font-size: 16px;'>
AI-powered OCT imaging is transforming retinal disease detection by utilizing deep learning models that can identify diseases like Diabetic Macular Edema (DME), 
Choroidal Neovascularization (CNV), and Drusen with high precision. The system not only accelerates the diagnostic process but also ensures that clinicians receive consistent and accurate results. 
With this technology, even subtle changes in the retina can be detected, enabling earlier intervention and more personalized treatment for patients.
</p>
""", unsafe_allow_html=True)

    st.markdown("""
<h2 style='color: black;'> Why OCT Retinal Analysis?</h2>

<p style='color: black; font-size: 16px;'>
Optical Coherence Tomography (OCT) is a non-invasive imaging technique that provides high-resolution cross-sectional images of the retina. 
It plays a crucial role in early diagnosis and treatment monitoring of diseases like Diabetic Macular Edema (DME), 
Choroidal Neovascularization (CNV), and Age-related Macular Degeneration (AMD).
This platform leverages AI to automate and accelerate the diagnostic process with accuracy and transparency, making it accessible for both clinical and research environments.
</p>
""", unsafe_allow_html=True)


    st.markdown("""
<h2 style='color: black;'> Key Features</h2>

<ul style='color: black; font-size: 16px;'>
    <li><b>AI-Powered Diagnosis:</b> Harnesses deep learning for accurate retinal disease classification.</li>
    <li><b>Fast Results:</b> Get real-time predictions with just one click.</li>
    <li><b>Explainable Insights:</b> Understand the visual signs associated with each disease prediction.</li>
    <li><b>User-Friendly:</b> Designed for clinicians, researchers, and students alike.</li>
</ul>
""", unsafe_allow_html=True)

    st.markdown("""<br>
    <footer style='color: black; text-align: center; padding: 20px; font-size: 14px;'>
        <p>&copy; 2025 OCT Retinal Analysis Platform. All Rights Reserved.</p>
        <p>Designed by ISE Students</p>
    </footer>
""", unsafe_allow_html=True)
# ---------------------- About Page ----------------------
elif app_mode == "About":
    st.markdown("""
    <style>
    .about-header {
        color: black;
        text-align: center;
    }
    .about-text {
        color: black;
        font-size: 16px;
        text-align: justify;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="stAboutPage">
    <h3 style='color: black; text-align: center;'>Overview</h3>
    <p style='color: black; font-size: 16px; text-align: justify;'>Our <strong>OCT Retinal Analysis Platform</strong> uses cutting-edge <strong>AI-driven technology</strong> to analyze <strong>OCT (Optical Coherence Tomography)</strong> images of the retina. This advanced imaging technique allows for the detection and diagnosis of a variety of retinal diseases at an early stage, significantly improving the chances of successful treatment.</p>
    <p style='color: black; font-size: 16px; text-align: justify;'>By leveraging deep learning algorithms and a large dataset of OCT images, our platform provides highly accurate, real-time predictions on retinal health, helping clinicians and healthcare professionals make informed decisions.</p>
    """, unsafe_allow_html=True)

    
    st.markdown("""<div class="stAboutPage">
    <p style='color: black; font-size: 16px; text-align: justify;'>
    <h3 style='color: black; text-align: center;'>Key Features</h3>
  <strong>"AI-Powered Detection</strong>: Utilizes deep learning models trained on thousands of OCT images to accurately detect retinal conditions."<br>
     <strong>Disease Detection</strong>: Capable of identifying conditions such as <strong>CNV (Choroidal Neovascularization)</strong>, 
    <br><strong>DME (Diabetic Macular Edema)</strong>, <strong>Drusen</strong>, and normal retinal conditions.
     <br><strong>User-Friendly Interface</strong>: Intuitive and easy-to-use platform for both medical professionals and patients.
    <br><strong>Real-Time Results</strong>: Immediate results upon uploading an OCT image, ensuring quick and efficient decision-making.
    <br><strong>Educational Insights</strong>: Provides users with detailed educational content about the detected conditions.
    <br><strong>Seamless Integration</strong>: Compatible with most OCT imaging devices for smooth integration into existing medical workflows.
    </p>
""", unsafe_allow_html=True)
    
    st.markdown("""<div class="stAboutPage">
    <h3 style='color: black; text-align: center;'>Technology Behind the Platform</h3>
    <p style='color: black; font-size: 16px; text-align: justify;'>
        Our platform is powered by <strong>deep learning</strong> and <strong>convolutional neural networks (CNNs)</strong>, trained on a vast amount of labeled OCT data. The neural network has learned 
        to identify subtle patterns and abnormalities in OCT scans that may be missed by the human eye. This ensures high accuracy in detecting a range of retinal diseases.
    </p>
        <strong>TensorFlow</strong> is used to build and deploy the AI models, while <strong>Streamlit</strong> powers the user-friendly interface for fast and interactive results. The platform is designed 
        to be fast, reliable, and scalable, ensuring seamless performance even with large volumes of data.
    <ul style='color: black; font-size: 16px; text-align: justify;'>
        <li><strong>Model Training</strong>: The model was trained on a diverse set of retinal images, making it robust and reliable.</li>
        <li><strong>TensorFlow Integration</strong>: For deploying and running the trained deep learning models.</li>
        <li><strong>Streamlit Interface</strong>: For easy image upload, display, and result visualization.</li>
    </ul></p>
""", unsafe_allow_html=True)
    
    
    st.markdown("""<div class="stAboutPage">
    <h3 style='color: black; text-align: center;'>Benefits of Early Detection</h3>
    <p style='color: black; font-size: 16px; text-align: justify;'>
        Early detection of retinal conditions is crucial for preventing <strong>vision loss</strong> and improving the overall prognosis of patients. 
        Our platform allows healthcare professionals to:
    <ul style='color: black; font-size: 16px; text-align: justify;'>
        <li><strong>Identify Retinal Diseases Early</strong>: Helps detect conditions before they progress to advanced stages, improving treatment outcomes.</li>
        <li><strong>Minimize Diagnostic Errors</strong>: Reduces the chances of human error in analyzing complex OCT scans.</li>
        <li><strong>Provide Actionable Insights</strong>: Delivers easy-to-understand recommendations for further medical intervention and patient care.</li>
        <li><strong>Enhance Patient Awareness</strong>: Educates patients on the importance of regular retinal checkups and early detection.</li>
    </ul></p>

""", unsafe_allow_html=True)
    

    st.markdown("""<div class="stAboutPage">
     <h3 style='color: black; text-align: center;'>Use Cases</h3>
    <ul style='color: black; font-size: 16px; text-align: justify;'>
        <li><strong>Clinics & Hospitals</strong>: Improve the accuracy and efficiency of retinal disease diagnosis.</li>
        <li><strong>Telemedicine</strong>: Enable remote diagnosis and consultations with minimal equipment.</li>
        <li><strong>Research Institutions</strong>: Aid in the analysis of OCT images for retinal research and development.</li>
        <li><strong>Patient Education</strong>: Provide patients with accessible information about their retinal health and possible treatment options.</li>
    </ul>
""", unsafe_allow_html=True)
    
    
    st.markdown("""<div class="stAboutPage">
     <h3 style='color: black; text-align: center;'>Data Privacy & Security</h3>
    <p style='color: black; font-size: 16px; text-align: justify;'>
        We prioritize the privacy and security of patient data. All images uploaded to the platform are processed securely, and no personal or sensitive information 
        is stored without explicit consent. The platform adheres to relevant healthcare regulations such as <strong>HIPAA</strong> (Health Insurance Portability and Accountability Act) 
        to ensure that your data remains safe and confidential.
    </p>
""", unsafe_allow_html=True)
    
    st.markdown("""<div class="stAboutPage">
     <h3 style='color: black; text-align: center;'>Future of Retinal Health</h3>
    <p style='color: black; font-size: 16px; text-align: justify;'>
        As technology continues to evolve, we aim to expand the capabilities of our platform, incorporating even more advanced techniques such as 
        <strong>3D OCT imaging</strong> and <strong>AI-based treatment recommendations</strong>. The future of retinal health is bright, and we’re excited to be at the forefront 
        of this technological revolution, improving the lives of patients around the world.
    </p>
""", unsafe_allow_html=True)
    st.markdown("""<br>
    <footer style='color: black; text-align: center; padding: 20px; font-size: 14px;'>
        <p>&copy; 2025 OCT Retinal Analysis Platform. All Rights Reserved.</p>
        <p>Designed by ISE Students</p>
    </footer>
""", unsafe_allow_html=True)



# ---------------------- Disease Identification Page ----------------------
elif app_mode == "Disease Identification":
    st.markdown("""
    <style>
    .disease-header {
        color: black;  /* Tomato red */
        text-align: center;
    }
    .disease-text {
        color: black;
        font-size: 16px;
        text-align: justify;
    }
    .disease-button {
        background: linear-gradient(to left, white, #306998);
        color: white;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='disease-header'> Disease Identification</h1>", unsafe_allow_html=True)

    # Image upload section
    st.markdown("<p class='disease-text'>Upload an OCT image to detect retinal diseases.</p>", unsafe_allow_html=True)

    test_image = st.file_uploader(" Upload an OCT Image", type=['jpg', 'jpeg', 'png'])
    if test_image is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=test_image.name) as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
        
        st.image(test_image, caption="Uploaded OCT Image", use_column_width=True)
    
    if st.button(" Predict", key="predict_button") and test_image is not None:
        with st.spinner(" Analyzing Image..."):
            result_index = model_prediction(temp_file_path)
            class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        
        st.success(f" Model Prediction: **{class_name[result_index]}**")
        # Display prediction details here...

    st.markdown("""<br>
    <footer style='color: black; text-align: center; padding: 20px; font-size: 14px;'>
        <p>&copy; 2025 OCT Retinal Analysis Platform. All Rights Reserved.</p>
        <p>Designed by ISE Students</p>
    </footer>
""", unsafe_allow_html=True)
        
