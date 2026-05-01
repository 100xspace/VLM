from trl import DPOTrainer
from transformers import TrainingArguments
training_args = TrainingArguments(
output_dir="./dpo_vlm",
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
learning_rate=5e-6,
num_train_epochs=2,
fp16=True,
logging_steps=10,
save_steps=500
)
dpo_trainer = DPOTrainer(
model=model,
ref_model=ref_model,
beta=0.1,                     # alignment strength
args=training_args,
train_dataset=dataset,
tokenizer=tokenizer
)
dpo_trainer.train()
