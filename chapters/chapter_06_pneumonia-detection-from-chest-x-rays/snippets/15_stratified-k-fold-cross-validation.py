from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs, cv_f1s, cv_sens = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df['image_id'], train_df['label'])):
    fold_train = train_df.iloc[train_idx]
    fold_val = train_df.iloc[val_idx]

    # Datasets/Loaders
    train_ds = PneumoniaDataset(fold_train, root_dir, get_transforms('train'))
    val_ds = PneumoniaDataset(fold_val, root_dir, get_transforms('val'))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Train fresh model (simplified; add your loop)
    model = PneumoniaDenseNet().to(device)
    # ... (optimizer, epochs via train_one_epoch)

    # Evaluate
    metrics = evaluate_model(model, val_loader, device)
    cv_aucs.append(metrics['AUC-ROC'])
    cv_f1s.append(metrics['F1'])
    cv_sens.append(metrics['Sensitivity (Recall)'])

    print(f"Fold {fold+1}: AUC={metrics['AUC-ROC']:.3f}, F1={metrics['F1']:.3f}")

print(f"CV AUC: {np.mean(cv_aucs):.3f} ± {np.std(cv_aucs):.3f}")
print(f"CV F1: {np.mean(cv_f1s):.3f} ± {np.std(cv_f1s):.3f}")
print(f"CV Sensitivity: {np.mean(cv_sens):.3f} ± {np.std(cv_sens):.3f}")
