import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2
# Import our custom classes from previous sections
# (Assuming these are saved in a local file named 'model_utils.py')
from model_utils import get_densenet_model, get_vlm_model, GradCAM, predict_vlm
# 1. Page Configuration
st.set_page_config(page_title="NeuroX: Pneumonia Detection", layout="wide")
# 2. Load Models (Cached)
@st.cache_resource
def load_specialist():
    # Load the trained DenseNet
    model = get_densenet_model()
    model.load_state_dict(torch.load("densenet_pneumonia.pth", map_location='cpu'))
    model.eval()
    return model
@st.cache_resource
def load_generalist():
    # Load the VLM (MedCLIP)
    model, processor = get_vlm_model()
    return model, processor
densenet = load_specialist()
vlm_model, vlm_processor = load_generalist()
