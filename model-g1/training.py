from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from model_g1 import ModelG1
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

# load dữ liệu
ds = load_dataset(
    "Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered",
    cache_dir="E:/datasets_dir"
)
num_samples = 64
response_samples = ds['train'].select(range(num_samples))['response']
instruction_samples = ds['train'].select(range(num_samples))['instruction']
noti=f"tổng lô dữ liệu là: {ds['train'].num_rows} nhưng chỉ dùng {num_samples}/{ds['train'].num_rows}"
[print(noti)] + [print("-", end="") for _ in range(len(noti))] + [print()]


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",
    cache_dir="E:/transformers_cache",
    token="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"
)
tokenizer.add_special_tokens({"bos_token": "<bos>"})

# mã hóa lô
def encode(batch, max_sequence_length, device, bos_token=True, eos_token=True):
    batch_encode = []
    for i in range(len(batch)):
        token = tokenizer.encode(batch[i], return_tensors="pt").to(device)[0].tolist()
        if bos_token: token = [tokenizer.bos_token_id] + token
        if eos_token: token.append(tokenizer.eos_token_id)
        if len(token) > max_sequence_length: token = token[:max_sequence_length]
        else:
            for _ in range(len(token), max_sequence_length): token.append(tokenizer.pad_token_id)
        batch_encode.append(token)
    batch_encode = torch.tensor(batch_encode, device=device)
    return batch_encode


# mã hóa các lô sang vector idx
max_sequence_length=20
src = encode(instruction_samples, max_sequence_length, device, bos_token=False, eos_token=False)
tgt = encode(response_samples, max_sequence_length, device, bos_token=True, eos_token=True)

# khởi tạo mô hình
model = ModelG1(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    device=device,
    max_sequence_length=max_sequence_length,
    vocab_size=tokenizer.vocab_size+1
)
# dự đoán thử
outputs = model(src[:1])
output_text_test = tokenizer.decode(torch.argmax(outputs, -1).tolist()[0])
noti=f"dự đoán đầu ra test của mô hình: {output_text_test}"
[print(noti)] + [print("-", end="") for _ in range(len(noti))] + [print()]
# in ra thông tin tham số mô hình
pytorch_total_params = sum(p.numel() for p in model.parameters())
noti=f"tổng tham số của mô hình là {pytorch_total_params}"
[print(noti)] + [print("-", end="") for _ in range(len(noti))] + [print()]


# training
from torch.utils.data import TensorDataset, DataLoader

# các tham số train
batch_size = 16
epochs = 20
lr = 0.001
model_saved_path = "./model.pt"

# chia dữ liệu
train_dataset = TensorDataset(src, tgt)
val_dataset = TensorDataset(src, tgt)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# các hàm tối ưu hóa
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# vòng lặp train
for epoch in range(epochs):
    model.train()
    total_loss = 0
    step_counter = 0
    for src_batch, tgt_batch in train_loader:
        (src_batch, tgt_batch) = (src_batch.long().to(device), tgt_batch.long().to(device))
        optimizer.zero_grad()

        y_pred = model(src_batch)
        loss = criterion(y_pred.reshape(-1, tokenizer.vocab_size+1), src_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        noti=f"bước: {step_counter}/{len(val_loader)} với mất mát: {loss.item()}"
        [print(noti)] + [print("_", end="") for _ in range(len(noti))] + [print()]
        step_counter += 1

    noti=f"vòng lặp: {epoch} hoàn thành tất cả bước với mất mát trung bình: {total_loss / len(val_loader)}"
    [print(noti)] + [print("_", end="") for _ in range(len(noti))] + [print()]
    print("saved model:", model_saved_path)
    # torch.save(model.state_dict(), model_saved_path)

    noti="dự đoán đầu ra thử"
    [print(noti)] + [print("-", end="") for _ in range(len(noti))] + [print()]
    model.eval()
    input_ids = encode(["Write a python script that accepts"], max_sequence_length, device, False, False)
    model_output = model(input_ids)
    output = torch.argmax(model_output, -1)
    noti=tokenizer.decode(output.tolist()[0])
    [print("y dự đoán:", noti)] + [print("-", end="") for _ in range(len(noti))] + [print()]
    noti=instruction_samples[:1]
    [print("y thực:", noti)] + [print("-", end="") for _ in range(len(noti[0]))] + [print()]