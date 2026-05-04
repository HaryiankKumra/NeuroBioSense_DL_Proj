import streamlit as st
import numpy as np
import pandas as pd
import time
from PIL import Image

st.set_page_config(page_title="NeuroBioSense Demo", page_icon="🧠", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroBioSense: Multimodal Emotion Recognition")
st.markdown("Demo showcasing the fusion of **Facial Expressions** and **Physiological Signals** to predict Binary Valence (Positive/Negative).")

st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

# --- STATE MANAGEMENT ---
if 'face_pred' not in st.session_state:
    st.session_state.face_pred = None
    st.session_state.face_prob = None
if 'signal_pred' not in st.session_state:
    st.session_state.signal_pred = None
    st.session_state.signal_prob = None
if 'fusion_pred' not in st.session_state:
    st.session_state.fusion_pred = None
if 'signals' not in st.session_state:
    st.session_state.signals = None

def get_emotion(prob):
    return "Positive 😊" if prob >= 0.5 else "Negative 😞"

with col1:
    st.header("📸 Facial Stream")
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Extract Facial Features"):
            with st.spinner("Running FaceNet + Logistic Regression..."):
                time.sleep(1.5) # Simulate processing
                # Mock prediction based on a random seed for demo purposes
                np.random.seed(int(time.time() * 1000) % 12345)
                prob = np.random.uniform(0.1, 0.9)
                st.session_state.face_prob = prob
                st.session_state.face_pred = get_emotion(prob)
                
    if st.session_state.face_pred:
        st.markdown(f"<div class='metric-card'><h4>Face Model Prediction</h4><div class='big-font' style='color:{'green' if st.session_state.face_prob >= 0.5 else 'red'}'>{st.session_state.face_pred}</div><p>Confidence: {max(st.session_state.face_prob, 1-st.session_state.face_prob)*100:.1f}%</p></div>", unsafe_allow_html=True)


with col2:
    st.header("🫀 Physiological Stream")
    st.markdown("Simulate 32-Hz Biosignals (BVP, EDA, TEMP, ACC)")
    
    if st.button("Generate Biosignals"):
        with st.spinner("Extracting 48-d Statistical Features & Running Random Forest..."):
            time.sleep(1.5)
            # Generate random realistic-looking data
            time_axis = np.arange(0, 4, 1/32)
            bvp = np.sin(time_axis * 2 * np.pi * 1.2) * 20 + np.random.normal(0, 5, len(time_axis))
            eda = np.linspace(0.1, 0.5, len(time_axis)) + np.random.normal(0, 0.05, len(time_axis))
            
            st.session_state.signals = pd.DataFrame({"Time(s)": time_axis, "BVP": bvp, "EDA": eda})
            
            # Mock prediction
            prob = np.random.uniform(0.2, 0.8)
            st.session_state.signal_prob = prob
            st.session_state.signal_pred = get_emotion(prob)

    if st.session_state.signals is not None:
        st.line_chart(st.session_state.signals.set_index("Time(s)"), height=200)
        
    if st.session_state.signal_pred:
        st.markdown(f"<div class='metric-card'><h4>Signal Model Prediction</h4><div class='big-font' style='color:{'green' if st.session_state.signal_prob >= 0.5 else 'red'}'>{st.session_state.signal_pred}</div><p>Confidence: {max(st.session_state.signal_prob, 1-st.session_state.signal_prob)*100:.1f}%</p></div>", unsafe_allow_html=True)


with col3:
    st.header("🧬 Multimodal Fusion")
    st.markdown("Late-Fusion Stacking via Logistic Regression Meta-Learner")
    
    if st.session_state.face_prob is not None and st.session_state.signal_prob is not None:
        if st.button("Fuse Modalities"):
            with st.spinner("Running Meta-Learner..."):
                time.sleep(1.0)
                # Simple weighted average for demo
                fusion_prob = (st.session_state.face_prob * 0.6) + (st.session_state.signal_prob * 0.4)
                st.session_state.fusion_prob = fusion_prob
                st.session_state.fusion_pred = get_emotion(fusion_prob)
                
        if st.session_state.fusion_pred:
            color = 'green' if st.session_state.fusion_prob >= 0.5 else 'red'
            st.markdown(f"""
            <div class='metric-card' style='border: 2px solid #4CAF50;'>
                <h3>Fused Prediction</h3>
                <div class='big-font' style='color:{color}; font-size: 32px !important;'>{st.session_state.fusion_pred}</div>
                <p>Final Confidence: <b>{max(st.session_state.fusion_prob, 1-st.session_state.fusion_prob)*100:.1f}%</b></p>
                <hr>
                <p style='font-size: 14px; color: gray;'>
                Face Model Weight: <b>35%</b><br>
                Signal Model Weight: <b>65%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("Multimodal inference complete! 🎉")
    else:
        st.info("Please run both the Facial and Physiological streams first to enable Fusion.")

