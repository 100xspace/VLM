Threshold tweaks: For max sensitivity, test threshold=0.3: sensitivities = [f1_score (all_labels, all_probs > t, pos_label=1) for t in np.linspace(0.1, 0.9, 20)].
