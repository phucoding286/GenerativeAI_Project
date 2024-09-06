import numpy as np

class CharToken:

    def __init__(self, x, y, maxlen_char=10, maxlen_word=20):
        self.maxlen_char = maxlen_char
        self.maxlen_word = maxlen_word
        self.word_to_idx = {"<pad>": 0, "<out>": 1, "<start>": 2, "<end>": 3}
        self.idx_to_word = {0: "<pad>", 1: "<out>", 2: "<start>", 3: "<end>"}
        self.x_batch_idx_word, self.y_batch_idx_word, self.word_to_idx, self.idx_to_word = self.get_tokenize_words(x, y)


    def get_tokenize_words(self, x, y):
        x_batch_idx_word, y_batch_idx_word = [], []
        idx_appeer = 4
        for sen_idx in range(len(x)):
            word_list = x[sen_idx].split() + y[sen_idx].split()
            for word_idx in range(len(word_list)):
                if word_list[word_idx] not in self.word_to_idx:
                    self.word_to_idx[word_list[word_idx]] = idx_appeer
                    self.idx_to_word[idx_appeer] = word_list[word_idx]
                    idx_appeer += 1
        for sen_idx in range(len(x)):
            word_list_x = x[sen_idx].split()[:self.maxlen_word]
            word_list_y = y[sen_idx].split()[:self.maxlen_word]
            x_batch = []
            y_batch = []
            for word_idx in range(self.maxlen_word):
                try:
                    x_batch.append(self.word_to_idx[word_list_x[word_idx]])
                except:
                    x_batch.append(self.word_to_idx["<pad>"])
            for word_idx in range(self.maxlen_word):
                try:
                    y_batch.append(self.word_to_idx[word_list_y[word_idx]])
                except:
                    if self.word_to_idx["<end>"] not in y_batch:
                        y_batch += [self.word_to_idx["<end>"]]
                    elif self.word_to_idx["<start>"] not in y_batch:
                        y_batch = [self.word_to_idx["<start>"]] + y_batch
                    else:
                        y_batch.append(self.word_to_idx["<pad>"])
            x_batch_idx_word.append(x_batch)
            y_batch_idx_word.append(y_batch)
        return np.array(x_batch_idx_word), np.array(y_batch_idx_word), self.word_to_idx, self.idx_to_word
    

    def word_encode(self, w):
        w_batch = []
        for sen_idx in range(len(w)):
            word_list = w[sen_idx].split()[:self.maxlen_word]
            batch = []
            for word_idx in range(len(word_list)):
                try:
                    batch.append(self.word_to_idx[word_list[word_idx]])
                except:
                    batch.append(self.word_to_idx["<out>"])
            for _ in range(len(w[sen_idx].split()), self.maxlen_word):
                batch.append(self.word_to_idx["<pad>"])
            w_batch.append(batch)
        return np.array(w_batch)
    

    def output_model_decode(self, output):
        output_sent = []
        for batch in output:
            sent = ""
            for word in batch:
                try:
                    sent += self.idx_to_word[int(word)]+" "
                except:
                    pass
            output_sent.append(sent.strip())
        return output_sent
