learning_rate = trial.suggest_loguniform("lr", 1e-6, 5e-4)
warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.1)
lora_rank = trial.suggest_categorical("lora_rank", [8, 16, 32, 64])
grad_accum = trial.suggest_categorical("grad_accum", [8, 16, 32])
beta = trial.suggest_float("beta", 0.05, 0.3)
    # --- Training arguments ---
training_args = TrainingArguments(
    output_dir=f"./optuna_trial_{trial.number}",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=grad_accum,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    num_train_epochs=1,
    fp16=True,
    logging_steps=50,
    save_steps=0,
    report_to="none"
)
    # --- DPO Trainer ---
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    beta=beta,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)
trainer.train()
    # --- Evaluation ---
pope_score = pope_eval(model, pope_dataset)
mmvet_score = mmvet_eval(model, mmvet_dataset)
    # Weighted score prioritizing hallucination resistance
final_score = 0.6 * pope_score + 0.4 * mmvet_score
    return final_score
