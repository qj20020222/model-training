device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Bacth_size = len()
def yield_tokens(train_data, tokenizer):
    for i, sample in enumerate(train_data):
        label,comment = sample
        yield tokenizer(comment)

model_name = "meta-llama/llama-2-long-13b-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def collate_fn(batch):
    traget = []
    token_index = []
    max_length =0
    for i, text in enumerate(batch):
        if len(tokens) > max_length:
            max_length = len(tokens)
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length = len(text))
        token_index.append(tokens)
    token_index = [index + [0]*(max_length-len(index)) for index in token_index]
    return (torch.tensor(token_index).to(torch.int32))


    
