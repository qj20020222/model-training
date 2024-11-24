from Dataprocesser import preprocess_function, preprocess_wikipedia_text
import Dataprocesser
from datasets import DatasetDict
from transformers import TrainingArguments, Traine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

input_dir = 'extracted'
processed_text = preprocess_wikipedia_text(input_dir)

model = "meta-llama/llama-2-long-13b-chinese"
tokenizer = AutoTokenizer.from_pretrained(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print some text
for i, text in enumerate(processed_text[:5]):
    print(f"Article {i+1}: {text[:100]}...")

tokenized_datasets = processed_text.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir="./llama2_long_zh",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=1000,
    num_train_epochs=3,
    logging_dir="./logs",
    save_steps=5000,
    max_seq_length=8192  # support long text
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets
)