from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch import nn
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# embedding mã hóa vị trí
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self, device):
        even_i = torch.arange(0, self.d_model, 2, device=device).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = torch.arange(self.max_sequence_length, device=device).reshape(self.max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, vocab_size, embedding_dim, dropout=0.1, padding_idx=0,
                 device: bool = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.positional_encoding_dropout = nn.Dropout(p=dropout)
        self.position = PositionalEncoding(embedding_dim, sequence_length)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx, device=self.device)
    

    def forward(self, x):
        x = self.embedding(x)
        position = self.position.forward(self.device)
        out = self.positional_encoding_dropout(x + position)
        return out.to(self.device)
    

# load dữ liệu
ds = load_dataset(
    "Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered",
    cache_dir="E:/datasets_dir"
)
num_samples = 15 # ds['train'].num_rows
response_samples = ds['train'].select(range(num_samples))['response']
instruction_samples = ds['train'].select(range(num_samples))['instruction']
print(f"tổng lô dữ liệu là: {num_samples}")
print("----------------------------")


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",
    cache_dir="E:/transformers_cache",
    token="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"
)
def encode(batch, max_sequence_length, device, bos_token=True, eos_token=True):
    batch_encode = []
    for i in range(len(batch)):
        token = tokenizer.encode(
            batch[i],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        ).to(device)[0].tolist()
        if bos_token: token = [tokenizer.bos_token_id] + token
        if eos_token: token.append(tokenizer.eos_token_id)
        if len(token) > max_sequence_length: token = token[:max_sequence_length]
        else:
            for _ in range(len(token), max_sequence_length): token.append(tokenizer.pad_token_id)
        batch_encode.append(token)
    batch_encode = torch.tensor(batch_encode, device=device)
    return batch_encode

def causal_mask(sequence_length, device):
    causal = nn.Transformer.generate_square_subsequent_mask(sequence_length)
    causal_bool = (causal == -torch.inf)
    return causal_bool.to(device)

def padding_mask(inputs, padding, device):
    return (inputs == padding).to(device)


# khởi tạo mô hình
class ModelT1(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_model = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation=nn.functional.gelu,
            batch_first=True,
            norm_first=True,
            device=device
        )
        self.position_embedding = PositionalEmbedding(
            sequence_length=20,
            vocab_size=tokenizer.vocab_size+1,
            embedding_dim=512,
            dropout=0.1,
            padding_idx=tokenizer.pad_token_id,
            device=device
        )
        self.linear_out = nn.Linear(
            in_features=512,
            out_features=tokenizer.vocab_size+1,
            device=device
        )
    def forward(self, x, y, src_causal_mask=None, src_padding_mask=None,
                tgt_causal_mask=None, tgt_padding_mask=None):
        x = self.position_embedding(x)
        y = self.position_embedding(y)
        out = self.transformer_model(
            src=x,
            tgt=y,
            src_mask=src_causal_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_is_causal=True
        )
        out = self.linear_out(out)
        return out
    
max_sequence_length=20
src = encode(instruction_samples, max_sequence_length, device, bos_token=False, eos_token=False)
tgt = encode(response_samples, max_sequence_length, device, bos_token=True, eos_token=True)

tgt_causal_mask = causal_mask(tgt.shape[1], device)
src_padding_mask = padding_mask(src[:1], tokenizer.pad_token_id, device)
tgt_padding_mask = padding_mask(tgt[:1], tokenizer.pad_token_id, device)

model = ModelT1()
outputs = model(
    src[:1],
    tgt[:1],
    tgt_causal_mask=tgt_causal_mask,
    tgt_padding_mask=tgt_padding_mask,
    src_padding_mask=src_padding_mask
)
output_text_test = tokenizer.decode(torch.argmax(outputs, -1).tolist()[0])
print(f"dự đoán đầu ra test của mô hình: {output_text_test}")
print("-----------------------------------------------------------")
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"tổng tham số của mô hình là {pytorch_total_params}")
print("------------------------------------")


from torch.utils.data import TensorDataset, DataLoader

batch_size = 1
epochs = 20
lr = 0.001
model_saved_path = "./model.pt"

train_dataset = TensorDataset(src, tgt)
val_dataset = TensorDataset(src, tgt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    step_counter = 0
    for src_batch, tgt_batch in train_loader:
        (src_batch, tgt_batch) = (src_batch.long().to(device), tgt_batch.long().to(device))
        optimizer.zero_grad()
        
        tgt_mask = causal_mask(tgt_batch.shape[1], device)
        src_padding_mask = padding_mask(src_batch, padding=tokenizer.pad_token_id, device=device)
        tgt_padding_mask = padding_mask(tgt_batch, padding=tokenizer.pad_token_id, device=device)

        y_pred = model(
            src_batch,
            tgt_batch,
            tgt_causal_mask=tgt_mask,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        loss = criterion(y_pred.reshape(-1, tokenizer.vocab_size+1), tgt_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"bước: {step_counter}/{len(val_loader)} với mất mát: {loss.item()}")
        step_counter += 1

    print(f"vòng lặp: {epoch} hoàn thành tất cả bước với mất mát trung bình: {total_loss / len(val_loader)}")
    print("saved model:", model_saved_path)
    torch.save(model.state_dict(), model_saved_path)

    print("dự đoán đầu ra thử")
    model.eval()
    input_ids = tokenizer.encode(["hello how are you?"], add_special_tokens=False)
    input_tgt = torch.tensor([[tokenizer.bos_token_id]+[0 for _ in range(max_sequence_length-1)]], device=device)
    model_output = model(input_ids, input_tgt)
    output = torch.argmax(model_output, -1)
    print(tokenizer.decode(torch.argmax(outputs, -1).tolist()[0]))