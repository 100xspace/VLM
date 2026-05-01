st.title("🩻 AI-Assisted Pneumonia Screening")
st.markdown("Upload a chest X-ray to detect signs of consolidation or opacity.")
uploaded_file = st.file_uploader("Drop X-Ray Here", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    # Convert file to Image
    image = Image.open(uploaded_file).convert("RGB")

    # Create two columns: Original vs. Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Scan")
        st.image(image, use_column_width=True)
    # --- INFERENCE BLOCK ---
    with col2:
        st.subheader("AI Analysis")

        if model_choice == "Specialist (DenseNet121)":
            # 1. Preprocess (Resize/Normalize)
            # (Assuming a preprocess function exists from Section 6.2)
            input_tensor = preprocess(image).unsqueeze(0)

            # 2. Predict
            with torch.no_grad():
                logit = densenet(input_tensor)
                prob = torch.sigmoid(logit).item()

            # 3. Explain (Grad-CAM)
            # We target the last dense block
            target_layer = densenet.features.denseblock4.denselayer16
            cam = GradCAM(densenet, target_layer)
            heatmap = cam.generate_heatmap(input_tensor)

            # Overlay
            # Convert PIL image to numpy for OpenCV
            img_np = np.array(image)
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

            # Blend based on slider
            final_display = cv2.addWeighted(img_np, 1 - heatmap_opacity, heatmap_colored, heatmap_opacity, 0)

        else:
            # VLM Approach
            prob, text_explanation = predict_vlm(vlm_model, vlm_processor, image)
            final_display = image # VLMs (without specialized attention hooks) show the original image

        # --- DISPLAY RESULTS ---

        # Color-coded confidence
        if prob > 0.5:
            st.error(f"**PNEUMONIA DETECTED** (Confidence: {prob:.1%})")
        else:
            st.success(f"**NORMAL** (Confidence: {1-prob:.1%})")

        # Progress bar visualization
        st.progress(prob)

        st.image(final_display, caption="AI Attention Map", use_column_width=True)
