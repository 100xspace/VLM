model = AutoModelForCausalLM.from_pretrained(
MODEL_NAME,
load_in_4bit=True,
device_map="auto",
bnb_4bit_compute_dtype=torch.float16,
bnb_4bit_use_double_quant=True
)
model = prepare_model_for_kbit_training(model)
qlora_config = LoraConfig(
r=64,
lora_alpha=16,
target_modules=["q_proj", "v_proj"],
lora_dropout=0.05,
bias="none",
task_type="CAUSAL_LM"
)
model = get_peft_model(model, qlora_config)
model.print_trainable_parameters()
