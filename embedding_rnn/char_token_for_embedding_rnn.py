import numpy as np

class CharToken:

    def __init__(self, x, y, maxlen_char=10, maxlen_word=20):
        self.maxlen_char = maxlen_char
        self.maxlen_word = maxlen_word
        self.char_dict = {}
        self.x_batch_idx_word, self.y_batch_idx_word, self.word_to_idx, self.idx_to_word = self.get_tokenize_words(x, y)
        self.x_batch_idx_char = self.get_tokenize_char(x, y)

    def get_tokenize_words(self, x, y):
        x_batch_idx_word, y_batch_idx_word = [], []
        word_to_idx, idx_to_word = {}, {}
        idx_appeer = 1
        for sen_idx in range(len(x)):
            word_list = x[sen_idx].split() + y[sen_idx].split()
            for word_idx in range(len(word_list)):
                if word_list[word_idx] not in word_to_idx:
                    word_to_idx[word_list[word_idx]] = idx_appeer
                    idx_to_word[idx_appeer] = word_list[word_idx]
                    idx_appeer += 1
        for sen_idx in range(len(x)):
            word_list_x = x[sen_idx].split()[:self.maxlen_word]
            word_list_y = y[sen_idx].split()[:self.maxlen_word]
            x_batch = []
            y_batch = []
            for word_idx in range(self.maxlen_word):
                try:
                    x_batch.append(word_to_idx[word_list_x[word_idx]])
                except:
                    x_batch.append(0)
            word_list = y[sen_idx].split()[:self.maxlen_word]
            for word_idx in range(self.maxlen_word):
                try:
                    y_batch.append(word_to_idx[word_list_y[word_idx]])
                except:
                    y_batch.append(0)
            x_batch_idx_word.append(x_batch)
            y_batch_idx_word.append(y_batch)
        return np.array(x_batch_idx_word), np.array(y_batch_idx_word), word_to_idx, idx_to_word
    

    def get_tokenize_char(self, x, y):
        x_batch_idx_char = []
        appeer_times = 1
        for sent_idx in range(len(x)):
            word_list = x[sent_idx].split() + y[sent_idx].split()
            for word_idx in range(len(word_list)):
                char_list = list(word_list[word_idx])
                for char in char_list:
                    if char not in self.char_dict:
                        self.char_dict[char] = appeer_times
                        appeer_times += 1
        for batch_idx in range(len(x)):
            word_batch = []
            for word_idx in range(self.maxlen_word):
                char_batch = []
                try:
                    word = x[batch_idx].split()[:self.maxlen_word][word_idx]
                    for char_idx in range(self.maxlen_char):
                        try:
                            char_batch.append(self.char_dict[list(word)[char_idx]])
                        except:
                            char_batch.append(0)
                except:
                    for char_idx in range(self.maxlen_char):
                        char_batch.append(0)
                word_batch.append(char_batch)
            x_batch_idx_char.append(word_batch)
        return np.array(x_batch_idx_char)
    

    def char_encode(self, x):
        x_batch_idx_char = []
        for batch_idx in range(len(x)):
            word_batch = []
            for word_idx in range(self.maxlen_word):
                char_batch = []
                try:
                    word = x[batch_idx].split()[:self.maxlen_word][word_idx]
                    for char_idx in range(self.maxlen_char):
                        try:
                            char_batch.append(self.char_dict[list(word)[char_idx]])
                        except:
                            char_batch.append(0)
                except:
                    for char_idx in range(self.maxlen_char):
                        char_batch.append(0)
                word_batch.append(char_batch)
            x_batch_idx_char.append(word_batch)
        return np.array(x_batch_idx_char)
    

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