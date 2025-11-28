import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DataCollatorForLanguageModeling

import os
os.environ["SAFETENSORS_FAST_GPU"] = "0"   # disable accelerated mmap
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


# Base model
# model_name = "mistralai/Mistral-7B-v0.1" # Too heavy for 8GB GPU
model_name = "microsoft/Phi-3-mini-4k-instruct"

# Load dataset
dataset = load_dataset("json", data_files="data/dataset.jsonl")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize(batch):
    texts = [
        instr + "\n" + resp
        for instr, resp in zip(batch["instruction"], batch["response"])
    ]
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=512
    )


# Tokenize dataset
tokenized = dataset.map(tokenize, batched=True)

# LoRA configuration
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     load_in_8bit=True
# )
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},          # fully CPU
    offload_folder="offload",        # <--- key part
    offload_state_dict=True,         # <--- huge RAM saver
    trust_remote_code=True
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    task_type=TaskType.CAUSAL_LM,
    lora_dropout=0.05,
)

model = get_peft_model(model, lora_cfg)

# Define LoRA config
args = TrainingArguments(
    output_dir="models/lora-output",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    save_strategy="epoch",
    learning_rate=2e-4,
)

collator = DataCollatorForLanguageModeling(
    tokenize,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    data_collator=collator,
)


trainer.train()
trainer.save_model("models/lora-output")
trainer.save_pretrained("models/lora-output")