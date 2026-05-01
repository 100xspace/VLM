for r in range(num_rounds):
        print(f"\nSelf-distillation round {r + 1}/{num_rounds}")
        # Clone current model as teacher
        teacher_model = copy.deepcopy(model); teacher_model.eval()
        # Train model (student) using itself as teacher
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        for batch in train_loader:
            images, input_ids, attention_mask = batch["images"], batch["input_ids"], batch["attention_mask"]
            # Teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(images, input_ids, attention_mask).logits
            # Student predictions
            student_logits = model(images, input_ids, attention_mask).logits
            # Distillation loss
            T = 2.0
            loss = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                            F.softmax(teacher_logits / T, dim=-1),
                            reduction="batchmean") * (T ** 2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Evaluate improvement
        accuracy = evaluate_model(model)
        print(f"Round {r + 1} accuracy: {accuracy:.2f}%")
    return model
