st.sidebar.title("Settings")
# Model Selection
model_choice = st.sidebar.radio(
    "Select AI Architect:",
    ("Specialist (DenseNet121)", "Generalist (MedCLIP)")
)
# Grad-CAM Intensity
heatmap_opacity = st.sidebar.slider("Explanation Opacity", 0.0, 1.0, 0.4)
st.sidebar.info(
    "Note: The Specialist is trained specifically on RSNA data. "
    "The Generalist uses Zero-Shot logic."
)
