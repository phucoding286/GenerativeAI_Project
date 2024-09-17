import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torch.utils.data import DataLoader, TensorDataset
from model_g1_tokenizer import ModelG1Tokenizer
from model_g1 import ModelG1


data = open("./model-g1/data.txt", "r", encoding="utf-8").read().splitlines()[:9000]
tokenizer = ModelG1Tokenizer(max_sequence_length=200, device=device)
tokenizer.vocab_init(data, [])
tokenizer.save("./model-g1/tokenizer.json")
tokenizer.load("./model-g1/tokenizer.json")
data = tokenizer.encode(data, start_token=True, end_token=True)

model = ModelG1(
    d_model=512,
    max_sequence_length=tokenizer.max_sequence_length,
    padding_idx=tokenizer.vocabulary[tokenizer.pad_token],
    vocab_size=data.max()+1,
    embed_dropout=0.01,
    nhead=8,
    num_layers=6,
    dim_ffn=1536,
    dropout=0.1,
    activation=torch.nn.functional.gelu,
    device=device
)


epochs = 20
model_saved_path = "./model-g1/model_g1.pt"
batch_size = 10

train_dataset = TensorDataset(data)
val_dataset = TensorDataset(data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.vocabulary[tokenizer.pad_token])
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    step_counter = 0
    for data_batch in train_loader:
        data_batch = data_batch[0].long().to(device)
        optimizer.zero_grad()

        y_pred = model(data_batch)
        loss = criterion(y_pred.reshape(-1, data.max()+1), data_batch.reshape(-1))

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(f"bước: {step_counter} với mất mát: {loss.item()}")
        step_counter += 1
    print(f"vòng lặp: {epoch} hoàn thành tất cả bước với mất mát trung bình: {total_loss / len(val_loader)}")
    print("saved model:", model_saved_path)
    torch.save(model.state_dict(), model_saved_path)

    print("dự đoán đầu ra thử")
    input_ids = tokenizer.encode(["I had a bad "], start_token=True, end_token=False)
    model_output = model(input_ids)
    output = torch.argmax(model_output, -1)
    print(tokenizer.decode(output.tolist(), False))