import os
import re
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import datasets
from transformers import TrainingArguments, Traine
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers import AutoTokenizer
from datasets import DatasetDict
from datasets import Dataset


tokenizer = AutoTokenizer.from_pretrained(model)

def clean_text(text):
    # remove HTML
    text = re.sub(r'<.*?>', '', text)
    # remove inference
    text = re.sub(r'\[\d+\]', '', text)
    # remove special mark
    text = re.sub(r'\[.*?\]', '', text)
    # remove tab
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            return clean_text(text)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def preprocess_wikipedia_text(input_dir):
    all_text = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    futures.append(executor.submit(preprocess_file, file_path))
        
        for future in futures:
            result = future.result()
            if result:
                all_text.append(result)
    
    return all_text

# Self-Supervised pretraining to get label
def getlabel(text):
    label_mapping = {0: "科技", 1: "历史", 2: "娱乐", 3: "体育"}
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=4)

    # 文章文本
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    # 获取预测标签
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    return predicted_label

def preprocess_function(examples):
    max_length =0
    for i, text in enumerate(examples):
        if len(text) > max_length:
            max_length = len(text)
    return tokenizer(examples, truncation=True, padding="max_length", max_length=max_length)

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

# maybe can save as data_dict to disk
def savedatadict (train_data, val_data):
    dataset_dict = DatasetDict({
    "train": Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    })
    dataset_dict.save_to_disk("data_dict.arrow")



    







    
