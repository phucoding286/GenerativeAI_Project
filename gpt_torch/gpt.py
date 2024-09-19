import torch
from torch import nn

# embedding mã hóa vị trí  
class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim,
                 dropout=0.1, padding_idx=0, device: bool = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        self.positional_encoding_dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
            device=device
        )

    def pos(self, max_sequence_length):
        even_i = torch.arange(
            start=0,
            end=self.embedding_dim,
            step=2,
            device=self.device
        ).float()
        denominator = torch.pow(
            10000,
            even_i/self.embedding_dim
        )
        position = torch.arange(
            max_sequence_length,
            device=self.device
        ).reshape(max_sequence_length, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack(
            [even_PE, odd_PE],
            dim=2
        )
        PE = torch.flatten(
            stacked,
            start_dim=1,
            end_dim=2
        )
        return PE.to(self.device)

    def forward(self, x):
        x = self.embedding(x)
        out = self.positional_encoding_dropout(x + self.pos(max_sequence_length=x.shape[1]))
        return out.to(self.device)
    

# khởi tạo mô hình
class GPT(nn.Module):
    "Mô hình tương tự GPT hoặc các mô hình tự hồi quy khác"
    """
    Khi training model này xin lưu ý, nên training nó theo kiểu tự hồi quy
    khác với các kiểu training truyền thống.

    Có nghĩa là mô hình sẽ dự đoán đầu ra dựa vào đầu vào, sau đó giảm loss
    dựa trên đầu vào và đầu ra, cách training này khiến mô hình học cách sinh
    văn bản tiếp theo.

    Vì vậy mô hình không cần training theo kiểu X hay Y truyền thống, với mô hình tự
    hồi quy, nó không thể training theo kiểu chuổi nguồn và chuổi đích.

    Nếu bạn muốn một mô hình dành cho dịch máy hãy quan tâm transformer.
    """
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1,
                activation=nn.functional.relu, batch_first=True, norm_first=True, device=None,
                vocab_size=None, pad_token_id=None):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.position_embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embedding_dim=d_model,
            dropout=dropout,
            padding_idx=pad_token_id,
            device=device
        )
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device
            ),
            num_layers=num_layers
        )
        self.linear_out = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device
        )

    def causal_mask(self, sequence_length, device):
        causal = nn.Transformer.generate_square_subsequent_mask(sequence_length)
        causal_bool = (causal == -torch.inf)
        return causal_bool.to(device)

    def create_key_padding_mask(self, inputs, device=None):
        key_padding_mask = (inputs == 0)
        return key_padding_mask.to(device)

    def forward(self, x):
        mask = self.causal_mask(
            x.shape[1],
            device=self.device
        )
        src_key_padding_mask = self.create_key_padding_mask(
            x,
            device=self.device
        )
        x = self.position_embedding(x)
        out = self.model(
            src=x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=True
        )
        out = self.linear_out(out)
        return out