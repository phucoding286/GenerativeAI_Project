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
num_samples = 100
response_samples = ds['train'].select(range(num_samples))['response']
instruction_samples = ds['train'].select(range(num_samples))['instruction']
print(f"tổng lô dữ liệu là: {ds['train'].num_rows} dùng {num_samples}/{ds['train'].num_rows}")


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",
    cache_dir="E:/transformers_cache",
    token="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"
)

# mã hóa các lô sang vector idx
samples = instruction_samples + response_samples
data = tokenizer.batch_encode_plus(
    instruction_samples,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=200,
    return_tensors="pt"
)['input_ids'].to(device)

# khởi tạo mô hình
model = ModelG1(
    d_model=512,
    nhead=8,
    num_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    device=device,
    vocab_size=tokenizer.vocab_size+1,
    pad_token_id=tokenizer.pad_token_id
)

# suy luận
def inference(inp, max_new_token=128):
    model.eval()
    input_ids = tokenizer.batch_encode_plus(
        [inp],
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=200,
        return_tensors="pt"
    )["input_ids"][0].tolist()
    [input_ids.append(tokenizer.pad_token_id) for _ in range(max_new_token)]
    input_ids = torch.tensor([input_ids]).to(device)
    model_output = model(input_ids)
    output = torch.argmax(model_output, -1)
    outputs = tokenizer.decode(output.tolist()[0])
    return outputs


# in ra thông tin tham số mô hình
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"tổng tham số của mô hình là {pytorch_total_params}")

# training
from torch.utils.data import TensorDataset, DataLoader

# các tham số train
batch_size = 16
epochs = 20
lr = 0.001
model_saved_path = "./model.pt"

# chia dữ liệu
train_dataset = TensorDataset(data)
val_dataset = TensorDataset(data)
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
    for batch in train_loader:
        batch = batch[0].long().to(device)
        optimizer.zero_grad()

        y_pred = model(batch)
        loss = criterion(y_pred.reshape(-1, tokenizer.vocab_size+1), batch.reshape(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"bước: {step_counter}/{len(val_loader)} với mất mát: {loss.item()}")
        step_counter += 1

    print(f"vòng lặp: {epoch} hoàn thành tất cả bước với mất mát trung bình: {total_loss / len(val_loader)}")

    print("saved model:", model_saved_path)
    torch.save(model.state_dict(), model_saved_path)

    print("y dự đoán:", inference("Write a python script that accepts a", max_new_token=data.shape[1]))
    print("y thực:", instruction_samples[:1])