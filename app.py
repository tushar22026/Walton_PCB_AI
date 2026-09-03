import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# page configuration
st.set_page_config(page_title="Walton AI Inspection", layout="wide", page_icon="🏭")

st.title(" Manufacturing Quality Control AI")
st.subheader("PCB Defect Detection System - Prototype for Walton")
st.markdown("---")

# Best model load 
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success(" Model loaded successfully!")
except Exception as e:
    st.sidebar.error(" Error loading model. Make sure 'best.pt' is in the same folder.")

st.sidebar.header("System Status")
st.sidebar.info("Model: YOLOv8 Nano\nmAP50: 98.9%\nTarget: Walton SMT Line")

# Image upload section
uploaded_file = st.file_uploader("Upload a PCB Image from the assembly line...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.info(" Original PCB Image")
        st.image(image, use_container_width=True)
    
    if st.button(" Run AI Inspection", type="primary"):
        with st.spinner('AI is inspecting PCB for defects...'):
            # YOLO prediction
            results = model.predict(image, conf=0.25)
            
            # Make result image (BGR to RGB)
            res_plotted = results[0].plot() 
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.success(" Inspection Complete!")
                st.image(res_rgb, caption='AI Inspection Result: Defects Marked', use_container_width=True)
                
            st.markdown("---")
            st.markdown("###  Inspection Summary & Analytics")
            
            # Count speed and defect
            inference_time = results[0].speed['inference']
            st.metric(label="⚡ Total AI Inference Time", value=f"{inference_time:.2f} ms")
            
            # List of defects found
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.warning(f" Total Defects Found: {len(boxes)}")
                class_names = results[0].names
                detected_classes = [class_names[int(cls)] for cls in boxes.cls]
                
                # Count of each defect type
                from collections import Counter
                counts = Counter(detected_classes)
                for defect, count in counts.items():
                    st.write(f"- **{defect.upper()}**: {count} unit(s)")
            else:
                st.success(" No defects found on this PCB! Passed Quality Control.")