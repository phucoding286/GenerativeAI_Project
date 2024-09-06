import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

def trainer(
        x_sequence,
        y_sequence, 
        model,
        vocab_size: int,
        epochs: int = 1,
        batch_size: int = 1,
        shuffle: bool = True,
        lr: float = 0.0001,
        ignore_index: int = 0,
        device: Optional[torch.device] = None
    ):
        train_dataset = TensorDataset(x_sequence.to(device), y_sequence.to(device))
        val_dataset = TensorDataset(x_sequence.to(device), y_sequence.to(device))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            step_counter = 0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                y_pred = model(x_batch, y_batch)
                loss = criterion(y_pred.reshape(-1, vocab_size), y_batch.long().reshape(-1))

                loss.backward()
                optimizer.step()
        
                total_loss += loss.item()
                print(f"STEP: {step_counter} with LOSS: {loss.item()}")
                step_counter += 1
            print(f"EPOCH: {epoch} completed all of steps current LOSS: {total_loss / len(val_loader)}")