import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier



class TokenizerML:

    def __init__(self, x_train, y_train, max_len_char=5, padding=20, num_models=6):
        self.max_len_char = max_len_char
        self.num_models = num_models
        self.padding = padding
        self.x_train = x_train
        self.y_train = y_train
        self.vocabuliary_dict_to_idx = {"<pad>": 0}
        self.vocabuliary_dict_idx_to = {0: "<pad>"}
        self.x_sequences_batch = []
        self.y_sequences_batch = []
        self.count = 1
        self.tokens = ["<start>", "<end>"]
        self.char_list = list("""`°–û1Čö23→45Ü678Û€9÷ü0-=qÖwertyuiop[]\\asdfghjk;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP}{ASDFGHJKL:"ZXCVBNM<>? éèẹẽẻêểễệếềýỵỳỷỹúùụũủứừựữửịỉĩìílóòọõỏốồộôỗổớờợỡởáàạãảấậầẫẩắằặẵẳâđÉÈẸẼẺÊỂỄỆẾỀÝỴỲỶỸÚÙỤŨỦỨỪỰỮỬỊỈĨÌÍLÓÒỌÕỎỐỒỘÔỖỔỚỜỢỠỞÁÀÂẠÃẢẤẬẦẪẨẮẰẶẴẲĐ""")
        self.punctions = list("""`°–ûČö→ÜÛ€÷ü-=Ö[]\\;',./~!@#$%^&*()_+}{:"<>?""")
        self.char_dict = {self.char_list[i]: i for i in range(len(self.char_list))}
        self.knn_vocab = self._fit_model_vocab()
        self._tokenizer()


    def _fit_model_vocab(self):
        (x_train, y_train) = self._preprocessing_data(self.x_train, self.y_train)
        (x_train, y_train) = self._get_word_list(x_train, y_train)
        knn = VotingClassifier(estimators=[(str(i), KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree", leaf_size=2)) for i in range(self.num_models)], voting="soft")
        knn.fit(x_train, y_train)
        return knn
    

    def _normalize_input_ML(self, x):
        x = self._preprocessing_data(x)
        x = self._sent_char_encode(x)
        return self.knn_vocab.predict(x)


    def _preprocessing_data(self, x_batch, y_batch=None):
        def remove_punction(txt: str):
            for punc in self.punctions:
                txt = txt.replace(punc, "")
            return txt.lower().strip()
        for i in range(len(x_batch)):
            x_batch[i] = remove_punction(x_batch[i])
            if y_batch is not None: y_batch[i] = remove_punction(y_batch[i])
        return (x_batch, y_batch) if y_batch is not None else x_batch


    def _get_word_list(self, x_batch, y_batch):
        vocabuliary = list(set(" ".join(x_batch + y_batch).split()))
        char_vocab_list = vocabuliary.copy()
        [self.char_list.append(c) if c not in self.char_list else 0 for c in list("".join(char_vocab_list))]
        for i in range(len(char_vocab_list)):
            char_vocab_list[i] = [self.char_dict[c] for c in list(char_vocab_list[i])]
            if len(char_vocab_list[i]) > self.max_len_char:
                char_vocab_list[i] = char_vocab_list[i][:self.max_len_char]
            else:
                for _ in range(len(list(char_vocab_list[i])), self.max_len_char): char_vocab_list[i].append(0)
        return (np.array(char_vocab_list), vocabuliary)
    

    def _sent_char_encode(self, x):
        vocabuliary = " ".join(x).split()
        char_vocab_list = vocabuliary.copy()
        for i in range(len(char_vocab_list)):
            char_vocab_list[i] = [self.char_dict[c] for c in list(char_vocab_list[i])]
            if len(char_vocab_list[i]) > self.max_len_char:
                char_vocab_list[i] = char_vocab_list[i][:self.max_len_char]
            else:
                for _ in range(len(list(char_vocab_list[i])), self.max_len_char): char_vocab_list[i].append(0)
        return np.array(char_vocab_list)
    


    def _tokenizer(self):
        vocabuliary = list(self._preprocessing_data(" ".join(self.x_train + self.y_train).split()))
        for vocab in vocabuliary:
            if vocab not in self.vocabuliary_dict_to_idx:
                self.vocabuliary_dict_idx_to[self.count] = vocab
                self.vocabuliary_dict_to_idx[vocab] = self.count
                self.count += 1
        for token in self.tokens:
            self.vocabuliary_dict_idx_to[self.count] = token
            self.vocabuliary_dict_to_idx[token] = self.count
            self.count += 1
        for batch_idx in range(len(self.x_train)):
            x_sequence = [self.vocabuliary_dict_to_idx[vocab] for vocab in self.x_train[batch_idx].split()]
            y_sequence = [self.vocabuliary_dict_to_idx[vocab] for vocab in self.y_train[batch_idx].split()]
            y_sequence = [self.vocabuliary_dict_to_idx['<start>']] + y_sequence + [self.vocabuliary_dict_to_idx['<end>']]
            if len(x_sequence) > self.padding:
                x_sequence = x_sequence[:self.padding]
            else:
                for _ in range(len(x_sequence), self.padding):
                    x_sequence.append(0)
            if len(y_sequence) > self.padding:
                y_sequence = y_sequence[:self.padding-1] + [self.vocabuliary_dict_to_idx['<end>']]
            else:
                for _ in range(len(y_sequence), self.padding):
                    y_sequence.append(0)
            self.x_sequences_batch.append(x_sequence)
            self.y_sequences_batch.append(y_sequence)
        self.x_sequences_batch = np.array(self.x_sequences_batch)
        self.y_sequences_batch = np.array(self.y_sequences_batch)

    
    def encode_input(self, x):
        def tokenizer(x):
            x_sequence = [self.vocabuliary_dict_to_idx[vocab] for vocab in x]
            if len(x_sequence) > self.padding:
                x_sequence = x_sequence[:self.padding]
            else:
                for _ in range(len(x_sequence), self.padding):
                    x_sequence.append(0)
            return np.array([x_sequence])
        x = self._normalize_input_ML(x)
        return tokenizer(x)
    

    def decode_output(self, x):
        sentence = ""
        for idx in x[0]: sentence += self.vocabuliary_dict_idx_to[int(idx)] + " "
        return sentence.strip()