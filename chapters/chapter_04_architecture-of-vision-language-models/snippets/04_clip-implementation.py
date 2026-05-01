import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.heatmap(similarity.cpu().numpy(), annot=True, fmt=".2f",
            xticklabels=["dog","cat"], yticklabels=["image"], cmap="YlGnBu")
plt.title("Text–Image Cosine Similarity Matrix")
plt.show()




Figure 4.16: Cosine similarity matrix
