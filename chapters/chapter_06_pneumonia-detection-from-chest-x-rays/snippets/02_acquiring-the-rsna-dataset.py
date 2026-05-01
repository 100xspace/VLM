# Step 2: Verify files
import os
import pandas as pd
print("Train images:", len(os.listdir("./rsna_data/stage_2_train_images/")))  # ~26,684
print("Test images:", len(os.listdir("./rsna_data/stage_2_test_images/")))    # ~3,000
print("Labels CSV shape:", pd.read_csv("./rsna_data/stage_2_train_labels.csv").shape) # (30,227, 6) including boxes
