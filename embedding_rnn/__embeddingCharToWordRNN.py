import keras
import tensorflow as tf


class EmbeddingRNN(keras.layers.Layer):

    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.char_list = """`1234567890-=qwertyuiop[]\\asdfghjk;'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP}{ASDFGHJKL:"ZXCVBNM<>? éèẹẽẻêểễệếềýỵỳỷỹúùụũủứừựữửịỉĩìíóòọõỏốồộỗổớờợỡởáàạãảấậầẫẩắằặẵẳđ"""
        self.char_size = len(list(self.char_list))
        self.embedding_dim = embedding_dim


    def build(self, input_shape):
        self.embedding = keras.layers.Embedding(input_dim=self.char_size, output_dim=self.embedding_dim, trainable=True)
        self.embedding_rnn = keras.layers.LSTM(units=self.embedding_dim, return_sequences=False, return_state=False, activation=None, use_bias=True)
        super(EmbeddingRNN, self).build(input_shape)
    

    def call(self, x):
        x = self.embedding(x)
        embed_rnn_op = self.embedding_rnn(tf.reshape(x, (-1, x.shape[2], x.shape[3])))
        embed_rnn_op = tf.reshape(embed_rnn_op, (-1, x.shape[1], embed_rnn_op.shape[-1]))
        return embed_rnn_op