from transformers import AutoModelForVision2Seq, AutoTokenizer
MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
MODEL_NAME,
torch_dtype=torch.float16,
device_map="auto"
)
# Reference model must be frozen
ref_model = AutoModelForVision2Seq.from_pretrained(
MODEL_NAME,
torch_dtype=torch.float16,
device_map="auto"
)
ref_model.eval()
for p in ref_model.parameters():
p.requires_grad = False

Dataset format:
 from datasets import Dataset
dataset = Dataset.from_list([
{
        "image": image_tensor,
        "prompt": "What brand is the laptop on the table?",
        "chosen": "The brand cannot be determined from the image.",
        "rejected": "It is an Apple MacBook Pro."
}
])
