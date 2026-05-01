model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            all_preds.extend((probs > threshold).astype(int))

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    auc_roc = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    prec, rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec, prec)
    PrecisionRecallDisplay(precision=prec, recall=rec).plot(ax=ax1)
    ax1.set_title(f'PR Curve (AUC-PR: {pr_auc:.3f})')

    fraction_of_positives, mean_predicted_value = calibration_curve(
        all_labels, all_probs, n_bins=10
    )
    ax2.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax2.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax2.set_title('Calibration Plot')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('eval_metrics.png')
    plt.close()

    return {
        'AUC-ROC': auc_roc,
        'F1': f1,
        'Sensitivity (Recall)': sensitivity,
        'Confusion Matrix': cm,
        'PR_AUC': pr_auc,
        'Calibration_ECE': np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    }

from torch.utils.data import DataLoader
val_dataset = PneumoniaDataset(val_df, root_dir, get_transforms('val'))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

metrics = evaluate_model(model, val_loader, device)
print(f"AUC: {metrics['AUC-ROC']:.3f}, F1: {metrics['F1']:.3f}, Sensitivity: {metrics['Sensitivity (Recall)']:.3f}")
print("Confusion Matrix:\n", metrics['Confusion Matrix'])

The trained model is evaluated on the validation set using performance metrics that assess discrimination, recall sensitivity, and classification balance, along with precision-recall behavior and probability calibration, as illustrated in the following output:
