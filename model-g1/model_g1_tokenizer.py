import torch
from collections import OrderedDict
import json


class ModelG1Tokenizer:

    def __init__(self, max_sequence_length: int = 200,
                 special_tokens: list = ["<pad>", "<start>", "<end>", "<out>"],
                 out_token: str = "<out>",
                 pad_token: str = "<pad>",
                 start_token: str = "<start>",
                 end_token: str = "<end>",
                 device: str = None,
                 dtype: str = None
                ):
        self.max_sequence_length = max_sequence_length
        self.special_tokens = special_tokens
        self.out_token = out_token
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.device = device
        self.dtype = dtype
    
    def vocab_init(self, x_: list, y_: list):
        self.vocabulary = {}
        for idx in range(len(self.special_tokens)): self.vocabulary[self.special_tokens[idx]] = idx
        sumary_characters = list(" ".join((x_ + y_)))
        get_vocabulary = OrderedDict().fromkeys(sumary_characters)
        for idx in range(len(get_vocabulary)): self.vocabulary[list(get_vocabulary)[idx]] = (idx+len(self.special_tokens))
        self.reverse_idx = {value: key for (key, value) in self.vocabulary.items()}

    def save(self, json_file: str = "vocabulary.json"):
        with open(json_file, "w", encoding="utf8") as file:
            json.dump([self.vocabulary, self.reverse_idx], file, ensure_ascii=False)

    def load(self, json_file: str = "vocabulary.json"):
        with open(json_file, "r", encoding="utf8") as file:
            data_loaded = json.load(file)
            self.vocabulary = data_loaded[0]
            self.reverse_idx = data_loaded[1]

    def encode(self, txt_batch: list, start_token=True, end_token=True):
        encoded = []
        for sentence in txt_batch:
            if start_token: batch_encode = [self.vocabulary[self.start_token]]
            else: batch_encode = []
            for char in list(sentence):
                try:
                    batch_encode.append(self.vocabulary[char])
                except:
                    batch_encode.append(self.vocabulary[self.out_token])
            if end_token: batch_encode.append(self.vocabulary[self.end_token])

            batch_encode_length = len(batch_encode)
            if batch_encode_length > self.max_sequence_length:
                batch_encode = batch_encode[:self.max_sequence_length]
            else:
                for _ in range(batch_encode_length, self.max_sequence_length):
                    batch_encode.append(self.vocabulary[self.pad_token])
            encoded.append(batch_encode)
        encoded = torch.tensor(encoded, dtype=self.dtype, device=self.device)
        return encoded.to(self.device)
    
    def decode(self, sequence_batch: list, skip_special_token: bool = True):
        text_batch = []
        for sequence in sequence_batch:
            text = ""
            for value in sequence:
                try:
                    text += self.reverse_idx[str(value)]
                except:
                    text += self.out_token
            text = text.replace(self.pad_token, "")
            if skip_special_token:
                for token in self.special_tokens:
                    text = text.replace(token, "")
            text_batch.append(text)
        return text_batch